//! End-to-end tests for pipelined terminal input, coalesced motion/resize, and batched
//! metadata/theme reads on high-latency connections.
use super::*;

#[allow(clippy::result_large_err)]
async fn accept_with_path(listener: &TcpListener) -> (String, WebSocketStream<TcpStream>) {
    let (tcp, _) = listener.accept().await.unwrap();
    let mut path = String::new();
    let ws = tokio_tungstenite::accept_hdr_async(
        tcp,
        |request: &tokio_tungstenite::tungstenite::handshake::server::Request,
         mut response: tokio_tungstenite::tungstenite::handshake::server::Response| {
            path = request.uri().path().to_owned();
            if let Some(protocol) = request.headers().get("Sec-WebSocket-Protocol") {
                response
                    .headers_mut()
                    .insert("Sec-WebSocket-Protocol", protocol.clone());
            }
            Ok(response)
        },
    )
    .await
    .unwrap();
    (path, ws)
}

fn empty_snapshot() -> Vec<u8> {
    let source = ghostty_vt::Terminal::new(ghostty_vt::TerminalOptions {
        cols: 20,
        rows: 4,
        max_scrollback: 1 << 12,
    })
    .unwrap();
    ghostty_vt::snapshot::encode(&source).unwrap().to_vec()
}

fn spawn_data_server(mut ws: WebSocketStream<TcpStream>) {
    tokio::spawn(async move {
        let Some(Ok(Message::Binary(hello))) = ws.next().await else {
            return;
        };
        if hello.len() != 17 || hello[0] != 1 {
            return;
        }
        let mut welcome = vec![0u8; 32];
        welcome[0] = 0x81;
        welcome[24] = 1;
        if ws.send(Message::Binary(welcome.into())).await.is_err() {
            return;
        }
        let _ = ws.send(Message::Binary(empty_snapshot().into())).await;
        // Keep the data socket open until the client closes.
        while let Some(Ok(frame)) = ws.next().await {
            if matches!(frame, Message::Close(_)) {
                break;
            }
        }
    });
}

fn key_payload(key: &str, text: &str, action: &str) -> Value {
    json!({
        "key": key,
        "mods": "",
        "consumed_mods": "",
        "text": text,
        "unshifted_codepoint": text.chars().next().map_or(0, |c| c as u32),
        "composing": false,
        "action": action,
    })
}

fn send_term(host: &HostConnection, block: &str, command: Command) {
    host.commands
        .send(HostCommand::Terminal {
            session: SESSION.into(),
            block: block.into(),
            command,
        })
        .unwrap();
}

async fn wait_pane_live(host: &HostConnection, block: &str) {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let live = {
                let m = host.model.lock().unwrap();
                m.terminals
                    .get(block)
                    .is_some_and(|t| t.model.lock().unwrap().connected)
            };
            if live {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("pane stream did not become live");
}

#[tokio::test]
async fn rapid_keystrokes_and_events_pipeline_without_waiting_for_prior_replies() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
    let (done_tx, done_rx) = tokio::sync::oneshot::channel();

    let server = tokio::spawn(async move {
        let (path, mut control) = accept_with_path(&listener).await;
        assert_eq!(path, "/api/control");
        welcome(&mut control, 21).await;

        // Accept the single data socket on a background task so control never blocks on it.
        let data_task = tokio::spawn(async move {
            let (path, ws) = accept_with_path(&listener).await;
            assert_eq!(path, "/api/data");
            spawn_data_server(ws);
        });

        let boot_req = recv(&mut control).await;
        assert_eq!(boot_req["method"], "session.bootstrap");
        let mut boot_val = serde_json::to_value(boot()).unwrap();
        boot_val["selected_session"]["view"] = json!({
            "session_id": SESSION,
            "label": "snapshot",
            "revision": 1,
            "windows": [{
                "window_id": "w1",
                "focused_block_id": "block:a",
                "layers": [{
                    "kind": "tiled",
                    "bounds": [0, 0, 1, 1],
                    "layout": {"block_id": "block:a"},
                    "blocks": [{
                        "block_id": "block:a",
                        "creator_name": activity::TERMINAL,
                        "owner": "client:test"
                    }]
                }]
            }]
        });
        boot_val["data_tickets"] = json!([{
            "session_id": SESSION,
            "block_id": "block:a",
            "data_ref_id": "terminal",
            "subprotocol": wire::PROTOCOL,
            "ticket": "AAAAAAAAAAAAAAAAAAAAAA",
            "expires_in_ms": 60_000
        }]);
        reply(&mut control, &boot_req, 1, boot_val).await;

        // Drain the 3 initial metadata queries for block:a.
        for _ in 0..3 {
            let req = recv(&mut control).await;
            reply(&mut control, &req, 2, json!({})).await;
        }
        ready_tx.send(()).unwrap();

        // Collect ALL 8 pipelined input requests BEFORE replying to a single one!
        // Under stop-and-wait this would deadlock until timeout.
        let mut held = Vec::with_capacity(8);
        for _ in 0..8 {
            let req = tokio::time::timeout(Duration::from_secs(2), recv(&mut control))
                .await
                .expect("all pipelined inputs must arrive without waiting for prior replies");
            held.push(req);
        }

        let summary: Vec<(&str, &str)> = held
            .iter()
            .map(|r| {
                let method = r["method"].as_str().unwrap();
                let detail = match method {
                    "client.key" => r["payload"]["key"]["action"].as_str().unwrap(),
                    "block.event" => r["payload"]["event"]["type"].as_str().unwrap(),
                    other => other,
                };
                (method, detail)
            })
            .collect();
        assert_eq!(
            summary,
            vec![
                ("client.key", "press"),
                ("client.key", "release"),
                ("client.key", "press"),
                ("client.key", "release"),
                ("client.key", "press"),
                ("client.key", "release"),
                ("block.event", "paste"),
                ("block.event", "wheel"),
            ]
        );
        assert_eq!(held[0]["payload"]["key"]["text"], "a");
        assert_eq!(held[2]["payload"]["key"]["text"], "b");
        assert_eq!(held[4]["payload"]["key"]["text"], "c");
        assert_eq!(held[6]["payload"]["event"]["payload"]["text"], "xyz");

        // Now reply to all 8 in FIFO order.
        for (i, req) in held.iter().enumerate() {
            reply(&mut control, req, 10 + i as u64, json!({"handled": false})).await;
        }
        // Confirm a subsequent ordinary request still works cleanly.
        let sentinel = recv(&mut control).await;
        assert_eq!(sentinel["method"], "test.sentinel");
        reply(&mut control, &sentinel, 20, json!({})).await;

        done_tx.send(()).unwrap();
        let _ = data_task.await;
    });

    let host = HostConnection::spawn(endpoint, None, None, None, true, None);
    connected(&host).await;
    ready_rx.await.unwrap();
    wait_pane_live(&host, "block:a").await;

    for (code, ch) in [("key_a", "a"), ("key_b", "b"), ("key_c", "c")] {
        send_term(
            &host,
            "block:a",
            Command::Key(key_payload(code, ch, "press")),
        );
        send_term(
            &host,
            "block:a",
            Command::Key(key_payload(code, "", "release")),
        );
    }
    send_term(&host, "block:a", Command::Paste("xyz".into()));
    send_term(
        &host,
        "block:a",
        Command::Wheel(json!({"x": 1, "y": 2, "delta_x": 0, "delta_y": -1})),
    );
    host.commands
        .send(HostCommand::Request {
            method: "test.sentinel".into(),
            payload: json!({}),
        })
        .unwrap();

    tokio::time::timeout(Duration::from_secs(5), done_rx)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(host.model.lock().unwrap().notice, None);
    assert!(host.model.lock().unwrap().connected);
    drop(host);
    tokio::time::timeout(Duration::from_secs(5), server)
        .await
        .unwrap()
        .unwrap();
}

#[test]
fn adjacent_mouse_motion_and_resize_coalesce_without_reordering_other_input() {
    let (tx, rx) = flume::unbounded();
    let mut deferred = VecDeque::new();
    let motion = |x: u16, button: Option<&str>, mods: &str| {
        let mut payload = json!({"action": "motion", "x": x, "y": 0, "mods": mods});
        if let Some(button) = button {
            payload["button"] = json!(button);
        }
        HostCommand::Terminal {
            session: SESSION.into(),
            block: "block:a".into(),
            command: Command::Mouse(payload),
        }
    };
    let resize = |cols: u16, rows: u16| HostCommand::Terminal {
        session: SESSION.into(),
        block: "block:a".into(),
        command: Command::Resize(cols, rows, 9, 18),
    };
    let press_key = |ch: &str| HostCommand::Terminal {
        session: SESSION.into(),
        block: "block:a".into(),
        command: Command::Key(key_payload("key_a", ch, "press")),
    };

    // 1. Consecutive motions with same button/mods collapse to the newest cell, stopping before a key.
    tx.send(motion(2, None, "")).unwrap();
    tx.send(motion(3, None, "")).unwrap();
    tx.send(press_key("a")).unwrap();
    tx.send(motion(4, Some("left"), "")).unwrap();
    tx.send(motion(5, Some("left"), "ctrl")).unwrap();
    tx.send(HostCommand::Terminal {
        session: SESSION.into(),
        block: "block:a".into(),
        command: Command::Mouse(
            json!({"action": "press", "button": "left", "x": 5, "y": 0, "mods": ""}),
        ),
    })
    .unwrap();
    tx.send(resize(80, 24)).unwrap();
    tx.send(resize(90, 25)).unwrap();
    tx.send(resize(100, 30)).unwrap();
    tx.send(press_key("b")).unwrap();

    let mut first = Command::Mouse(json!({"action": "motion", "x": 1, "y": 0, "mods": ""}));
    coalesce_terminal_command(SESSION, "block:a", &mut first, &mut deferred, &rx);
    match first {
        Command::Mouse(p) => assert_eq!(p["x"], 3, "motions 1..=3 collapsed to 3"),
        _ => panic!("expected mouse motion"),
    }

    // Next command in deferred must be the untouched key 'a'.
    match deferred.pop_front().unwrap() {
        HostCommand::Terminal {
            command: Command::Key(k),
            ..
        } => assert_eq!(k["text"], "a"),
        _ => panic!("expected key 'a' next"),
    }

    // Motion with button="left", mods="" must NOT collapse into motion with mods="ctrl".
    let HostCommand::Terminal {
        mut command,
        session,
        block,
    } = rx.try_recv().unwrap()
    else {
        panic!("expected motion 4");
    };
    coalesce_terminal_command(&session, &block, &mut command, &mut deferred, &rx);
    match command {
        Command::Mouse(p) => assert_eq!((p["x"].as_u64(), p["mods"].as_str()), (Some(4), Some(""))),
        _ => panic!("expected motion 4"),
    }

    // Motion with mods="ctrl" must NOT collapse into a mouse press.
    let HostCommand::Terminal {
        mut command,
        session,
        block,
    } = deferred.pop_front().unwrap()
    else {
        panic!("expected motion 5");
    };
    coalesce_terminal_command(&session, &block, &mut command, &mut deferred, &rx);
    match command {
        Command::Mouse(p) => assert_eq!(p["x"], 5),
        _ => panic!("expected motion 5"),
    }
    match deferred.pop_front().unwrap() {
        HostCommand::Terminal {
            command: Command::Mouse(p),
            ..
        } => assert_eq!(p["action"], "press"),
        _ => panic!("expected mouse press next"),
    }

    // Three adjacent resizes collapse to the final size, leaving key 'b' queued after.
    let HostCommand::Terminal {
        mut command,
        session,
        block,
    } = rx.try_recv().unwrap()
    else {
        panic!("expected first resize");
    };
    coalesce_terminal_command(&session, &block, &mut command, &mut deferred, &rx);
    assert!(matches!(command, Command::Resize(100, 30, 9, 18)));
    match deferred.pop_front().unwrap() {
        HostCommand::Terminal {
            command: Command::Key(k),
            ..
        } => assert_eq!(k["text"], "b"),
        _ => panic!("expected key 'b' after coalesced resize"),
    }
}

#[tokio::test]
async fn pipelined_input_api_error_notifies_without_disconnecting() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
    let (done_tx, done_rx) = tokio::sync::oneshot::channel();

    let server = tokio::spawn(async move {
        let (_, mut control) = accept_with_path(&listener).await;
        welcome(&mut control, 21).await;
        let data_task = tokio::spawn(async move {
            let (_, ws) = accept_with_path(&listener).await;
            spawn_data_server(ws);
        });
        let boot_req = recv(&mut control).await;
        let mut boot_val = serde_json::to_value(boot()).unwrap();
        boot_val["selected_session"]["view"] = json!({
            "session_id": SESSION,
            "label": "snapshot",
            "revision": 1,
            "windows": [{
                "window_id": "w1",
                "focused_block_id": "block:a",
                "layers": [{
                    "kind": "tiled",
                    "bounds": [0, 0, 1, 1],
                    "layout": {"block_id": "block:a"},
                    "blocks": [{
                        "block_id": "block:a",
                        "creator_name": activity::TERMINAL,
                        "owner": "client:test"
                    }]
                }]
            }]
        });
        boot_val["data_tickets"] = json!([{
            "session_id": SESSION,
            "block_id": "block:a",
            "data_ref_id": "terminal",
            "subprotocol": wire::PROTOCOL,
            "ticket": "AAAAAAAAAAAAAAAAAAAAAA",
            "expires_in_ms": 60_000
        }]);
        reply(&mut control, &boot_req, 1, boot_val).await;
        for _ in 0..3 {
            let req = recv(&mut control).await;
            reply(&mut control, &req, 2, json!({})).await;
        }
        ready_tx.send(()).unwrap();

        let first = recv(&mut control).await;
        let second = recv(&mut control).await;
        assert_eq!(first["method"], "client.key");
        assert_eq!(second["method"], "client.key");
        send(
            &mut control,
            json!({
                "type": "response",
                "id": first["id"],
                "seq": 3,
                "error": {"code": "bad_request", "status": 400, "message": "lua handler failed"}
            }),
        )
        .await;
        reply(&mut control, &second, 4, json!({"handled": false})).await;

        let third = recv(&mut control).await;
        assert_eq!(third["method"], "client.key");
        assert_eq!(third["payload"]["key"]["text"], "c");
        reply(&mut control, &third, 5, json!({"handled": false})).await;

        done_tx.send(()).unwrap();
        let _ = data_task.await;
    });

    let host = HostConnection::spawn(endpoint, None, None, None, true, None);
    connected(&host).await;
    ready_rx.await.unwrap();
    wait_pane_live(&host, "block:a").await;
    let epoch = host.model.lock().unwrap().connection_epoch;

    send_term(
        &host,
        "block:a",
        Command::Key(key_payload("key_a", "a", "press")),
    );
    send_term(
        &host,
        "block:a",
        Command::Key(key_payload("key_b", "b", "press")),
    );
    until(&host, |m| {
        m.notice
            .as_deref()
            .is_some_and(|n| n.contains("lua handler failed"))
    })
    .await;
    assert!(host.model.lock().unwrap().connected);
    assert_eq!(host.model.lock().unwrap().connection_epoch, epoch);

    send_term(
        &host,
        "block:a",
        Command::Key(key_payload("key_c", "c", "press")),
    );
    tokio::time::timeout(Duration::from_secs(5), done_rx)
        .await
        .unwrap()
        .unwrap();
    drop(host);
    tokio::time::timeout(Duration::from_secs(5), server)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn query_activity_pipelines_multi_pane_metadata_in_one_round_trip() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());

    let server = tokio::spawn(async move {
        let mut ws = accept(&listener, "rex.control.v1").await;
        welcome(&mut ws, 21).await;
        let boot_req = recv(&mut ws).await;
        let mut boot_val = serde_json::to_value(boot()).unwrap();
        boot_val["selected_session"]["view"] = json!({
            "session_id": SESSION,
            "label": "snapshot",
            "revision": 1,
            "windows": [{
                "window_id": "w1",
                "layers": [{
                    "kind": "tiled",
                    "layout": {"block_id": "block:a"},
                    "blocks": [
                        {"block_id": "block:a", "creator_name": activity::TERMINAL},
                        {"block_id": "block:b", "creator_name": activity::TERMINAL},
                        {"block_id": "block:c", "creator_name": activity::TERMINAL}
                    ]
                }]
            }]
        });
        reply(&mut ws, &boot_req, 1, boot_val).await;

        // Receive ALL 9 metadata requests before sending ANY reply!
        let mut reqs = Vec::with_capacity(9);
        for _ in 0..9 {
            let req = tokio::time::timeout(Duration::from_secs(2), recv(&mut ws))
                .await
                .expect("all 9 metadata requests must be pipelined");
            reqs.push(req);
        }
        let expected_order: Vec<(&str, &str)> = ["block:a", "block:b", "block:c"]
            .into_iter()
            .flat_map(|b| {
                [
                    ("com.superlogical.terminal.program_status", b),
                    ("com.superlogical.terminal.title", b),
                    ("com.superlogical.terminal.process", b),
                ]
            })
            .collect();
        let actual_order: Vec<(&str, &str)> = reqs
            .iter()
            .map(|r| {
                (
                    r["method"].as_str().unwrap(),
                    r["payload"]["block_id"].as_str().unwrap(),
                )
            })
            .collect();
        assert_eq!(actual_order, expected_order);

        for (i, req) in reqs.iter().enumerate() {
            let block = req["payload"]["block_id"].as_str().unwrap();
            let payload = match req["method"].as_str().unwrap() {
                "com.superlogical.terminal.title" => json!({"title": format!("title-{block}")}),
                "com.superlogical.terminal.process" => json!({"foreground": {"name": "vim"}}),
                _ => json!({"records": []}),
            };
            reply(&mut ws, req, 2 + i as u64, payload).await;
        }
        assert!(matches!(ws.next().await, Some(Ok(Message::Close(_)))));
    });

    let host = HostConnection::spawn(endpoint, None, None, None, false, None);
    until(&host, |m| {
        m.title("block:a") == Some("title-block:a")
            && m.title("block:b") == Some("title-block:b")
            && m.title("block:c") == Some("title-block:c")
            && m.programs.get("block:c").map(String::as_str) == Some("vim")
    })
    .await;
    drop(host);
    tokio::time::timeout(Duration::from_secs(5), server)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn declare_themes_probes_once_then_pipelines_remaining_blocks() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());

    let server = tokio::spawn(async move {
        let (path, mut control) = accept_with_path(&listener).await;
        assert_eq!(path, "/api/control");
        welcome(&mut control, 21).await;
        // Drain data-socket connection attempts quietly so they don't stall accept.
        let data_drain = tokio::spawn(async move {
            loop {
                let (_, _ws) = accept_with_path(&listener).await;
            }
        });

        let boot_req = recv(&mut control).await;
        let mut boot_val = serde_json::to_value(boot()).unwrap();
        boot_val["selected_session"]["view"] = json!({
            "session_id": SESSION,
            "label": "snapshot",
            "revision": 1,
            "windows": [{
                "window_id": "w1",
                "layers": [{
                    "kind": "tiled",
                    "layout": {"block_id": "block:a"},
                    "blocks": [
                        {"block_id": "block:a", "creator_name": activity::TERMINAL},
                        {"block_id": "block:b", "creator_name": activity::TERMINAL},
                        {"block_id": "block:c", "creator_name": activity::TERMINAL}
                    ]
                }]
            }]
        });
        boot_val["data_tickets"] = json!([
            {"session_id": SESSION, "block_id": "block:a", "data_ref_id": "terminal", "subprotocol": wire::PROTOCOL, "ticket": "AAAAAAAAAAAAAAAAAAAAAA", "expires_in_ms": 60_000},
            {"session_id": SESSION, "block_id": "block:b", "data_ref_id": "terminal", "subprotocol": wire::PROTOCOL, "ticket": "AAAAAAAAAAAAAAAAAAAAAA", "expires_in_ms": 60_000},
            {"session_id": SESSION, "block_id": "block:c", "data_ref_id": "terminal", "subprotocol": wire::PROTOCOL, "ticket": "AAAAAAAAAAAAAAAAAAAAAA", "expires_in_ms": 60_000}
        ]);
        reply(&mut control, &boot_req, 1, boot_val).await;

        // First set_theme is a single probe when theme_declared is still false.
        let probe = recv(&mut control).await;
        assert_eq!(probe["method"], "com.superlogical.terminal.set_theme");
        reply(&mut control, &probe, 2, json!({})).await;

        // Remaining 2 set_theme calls must arrive pipelined before replying to either.
        let second = tokio::time::timeout(Duration::from_secs(2), recv(&mut control))
            .await
            .expect("second set_theme");
        let third = tokio::time::timeout(Duration::from_secs(2), recv(&mut control))
            .await
            .expect("third set_theme must pipeline with second");
        assert_eq!(second["method"], "com.superlogical.terminal.set_theme");
        assert_eq!(third["method"], "com.superlogical.terminal.set_theme");
        reply(&mut control, &second, 3, json!({})).await;
        reply(&mut control, &third, 4, json!({})).await;

        // Answer initial 9 metadata queries.
        for i in 0..9 {
            let req = recv(&mut control).await;
            reply(&mut control, &req, 5 + i, json!({})).await;
        }

        // Now switch theme: theme_declared is already true, so all 3 must pipeline together!
        let mut batch = Vec::with_capacity(3);
        for _ in 0..3 {
            let req = tokio::time::timeout(Duration::from_secs(2), recv(&mut control))
                .await
                .expect("all 3 theme updates must pipeline");
            assert_eq!(req["method"], "com.superlogical.terminal.set_theme");
            batch.push(req);
        }
        for (i, req) in batch.iter().enumerate() {
            reply(&mut control, req, 20 + i as u64, json!({})).await;
        }
        let sentinel = recv(&mut control).await;
        assert_eq!(sentinel["method"], "test.sentinel");
        reply(&mut control, &sentinel, 30, json!({})).await;
        data_drain.abort();
    });

    let host = HostConnection::spawn(
        endpoint,
        None,
        None,
        None,
        true,
        Some(SetTheme::new().background("#111111")),
    );
    connected(&host).await;
    until(&host, |m| m.theme_declared).await;
    host.commands
        .send(HostCommand::Theme(Some(
            SetTheme::new().background("#222222"),
        )))
        .unwrap();
    host.commands
        .send(HostCommand::Request {
            method: "test.sentinel".into(),
            payload: json!({}),
        })
        .unwrap();
    tokio::time::timeout(Duration::from_secs(5), server)
        .await
        .unwrap()
        .unwrap();
    drop(host);
}
