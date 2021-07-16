use std::time::{Duration};
use std::net::TcpStream;
use std::sync::{Arc, Mutex};
use std::thread;
use std::cmp::min;
use websocket::{Message, OwnedMessage};
use websocket::sync::{Client, Server};
use serde_json::{Value, json};
use crate::game::{Command, Game};
use crate::generation::{generate_areas, get_polygons};

fn game_loop(mut players: Vec<Client<TcpStream>>) {
    let players_count = players.len();
    let (areas, graph) = generate_areas(players_count * 10, 10..12);
    
    let mut game;
    {
        let eliminate_every_n_round = 3;
        let mut config = json!({
            "areas": get_polygons(areas),
            "graph": graph,
            "eliminate_every_n_round": eliminate_every_n_round,
        });
    
        game = Game::new(players_count, eliminate_every_n_round, graph);
        
        for (player_id, player) in players.iter_mut().enumerate() {
            config["me"] = Value::from(player_id);
            if let Err(e) = player.send_message(&Message::text(config.to_string())) {
                println!("Send error: {:?}", e);
            }
        }
    }

    while !game.is_ended() {
        let state = &Message::text(serde_json::to_string(&game).unwrap());
        for player in &mut players {
            if let Err(e) = player.send_message(state) {
                println!("Send error: {:?}", e);
            }
        }
        
        {
            let player_id = game.get_current_player();
            let player = &mut players[player_id];

            let command = match player.recv_message() {
                Ok(OwnedMessage::Text(raw_data)) => {
                    serde_json::from_str(raw_data.as_str()).unwrap_or(Command::EndTurn)
                }
                _ => {Command::EndTurn},
            };

            println!("Player {} turn: {:?}", player_id, command);
            let _ = game.turn(command);
        }
    }

    println!("Match ended");
}

fn match_scheduler(clients: Arc<Mutex<Vec<Client<TcpStream>>>>) {
    loop {
        thread::sleep(Duration::from_secs(5));
        let mut acquired_clients = clients.lock().unwrap();
        let clients_count = acquired_clients.len();
        if clients_count > 1 {
            let players = acquired_clients
                .drain(..min(clients_count, 8))
                .collect();
            
            thread::spawn(move || game_loop(players));
            println!("Match started");
        }
    }
}

pub fn connection_handler() {
    let server = Server::bind("0.0.0.0:9001").unwrap();

    let clients = Arc::new(Mutex::new(Vec::new()));
    let clients_clone = clients.clone();
    thread::spawn(move || match_scheduler(clients_clone));
    
    // let mut next_client_id: usize = 0;

    for connection in server.filter_map(Result::ok) {
        let client = connection.accept().unwrap();
        println!("New connection: {}", client.peer_addr().unwrap());
        let mut aquired_clients = clients.lock().unwrap();
        aquired_clients.push(client);
        // next_client_id += 1;
    }
}
