// use std::time::{Duration};
use std::net::TcpStream;
use log::info;
use websocket::server::NoTlsAcceptor;
use websocket::{Message, OwnedMessage};
use websocket::sync::{Client, Server};
use serde_json::{Value, json};
use crate::game::{Command, Game};
use crate::generation::{generate_areas, get_polygons};

pub fn game_loop(players: Vec<Client<TcpStream>>) {
    let mut players: Vec<_> = players.into_iter().map(|player| Some(player)).collect();
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
        info!("{}", serde_json::to_string(&config).unwrap());
        
        game = Game::new(players_count, eliminate_every_n_round, graph);
        
        for (player_id, player) in players.iter_mut().enumerate() {
            config["me"] = Value::from(player_id);
            if let Err(e) = player.as_mut().unwrap().send_message(&Message::text(json!({"start": config}).to_string())) {
                info!("Send error: {:?}", e);
            }
        }
    }

    let game_end_message = &Message::text(json!({"end": {}}).to_string());
    let close_message = &Message::close_because(1000, "Eliminated");
    
    while !game.is_ended() {
        info!("{}", serde_json::to_string(&game).unwrap());

        let state = &Message::text(json!({"state": game}).to_string());
        
        players
            .iter_mut()
            .filter(|player| player.is_some())
            .map(|player| player.as_mut().unwrap())
            .for_each(|player| {
                if let Err(e) = player.send_message(state) {
                    info!("Send error: {:?}", e);
                }
            });
        
        {
            let player_id = game.get_current_player();
            let wrapped_player = &mut players[player_id];
            let mut close = false;
            let command = match wrapped_player {
                Some(player) => {
                    match player.recv_message() {
                        // TODO: EndTurn only if close or error, if ping then skip
                        Ok(OwnedMessage::Text(raw_data)) => {
                            serde_json::from_str(raw_data.as_str()).unwrap_or(Command::EndTurn)
                        }
                        Ok(OwnedMessage::Close(_)) => {
                            close = true;
                            Command::EndTurn
                        }
                        _ => {
                            // TODO: make this player None
                            Command::EndTurn
                        },
                    }
                }
                None => Command::EndTurn
            };

            if close {
                *wrapped_player = None
            }
            
            info!("{}", json!({"player": player_id, "command": command}).to_string());
            let _ = game.turn(command);
        }

        // Return eliminated players to matchmaking pool
        let mut eliminated = vec![true; players_count];
        for player_id in game.get_players() {
            eliminated[player_id] = false;
        }
        
        players = eliminated
            .into_iter()
            .zip(players.into_iter())
            .map(|(eliminated, player)| {
                if eliminated {
                    if let Some(mut client) = player {
                        // TODO: Make dedicated function that handles errors
                        if let Err(e) = client.send_message(game_end_message) {
                            info!("Send error: {:?}", e);
                        }
                        if let Err(e) = client.send_message(close_message) {
                            info!("Send error: {:?}", e);
                        }
                        return None;
                    }
                }

                player
            })
            .collect();
    }

    // TODO: make game end function    
    players
        .into_iter()
        .for_each(|player| {
            if let Some(mut client) = player {
                // TODO: Make dedicated function that handles errors
                if let Err(e) = client.send_message(game_end_message) {
                    info!("Send error: {:?}", e);
                }
                if let Err(e) = client.send_message(close_message) {
                    info!("Send error: {:?}", e);
                }
            }
        });

    info!("{}", json!({"winners": game.get_players()}).to_string());
}


pub fn connection_handler(server: Server<NoTlsAcceptor>, number_of_clients: usize) -> Vec<Client<TcpStream>>{
    let mut clients = Vec::with_capacity(number_of_clients);
    
    for connection in server.filter_map(Result::ok) {
        let client = connection.accept().unwrap();
        info!("Connection: {}", client.peer_addr().unwrap());

        clients.push(client);

        if clients.len() == number_of_clients {
            break;
        }
    }

    clients
}
