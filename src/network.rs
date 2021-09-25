use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use std::thread;
use std::net::TcpStream;
use log::{info, debug};
use websocket::server::NoTlsAcceptor;
use websocket::{Message, OwnedMessage};
use websocket::sync::{Client, Server, Writer};
use serde_json::{Value, json};
use crate::game::{Command, Game, GameConfig, get_polygons};


enum CommandError {
    ReadTimeoutError,
    ParseError,
    Disconnected,
}

struct PlayerCommunicator {
    writer: Writer<TcpStream>,
    queue: Arc<Mutex<VecDeque<Result<(Command, Instant), CommandError>>>>,
    message_event: Arc<Condvar>,
    disconnected: Arc<AtomicBool>,
}

impl PlayerCommunicator {
    fn new(player: Client<TcpStream>) -> PlayerCommunicator {
        let (mut reader, writer) = player.split().unwrap();

        let queue = Arc::new(Mutex::new(VecDeque::new()));
        let queue_copy = queue.clone();

        let message_event = Arc::new(Condvar::new());
        let message_event_copy = message_event.clone();

        let disconnected = Arc::new(AtomicBool::new(false));
        let disconnected_copy = disconnected.clone();
        
        thread::spawn(move || {
            loop {
                let message = reader.recv_message();
                
                if let Ok(OwnedMessage::Close(_)) | Err(_) = message {
                    disconnected_copy.store(true, Ordering::Relaxed);
                    message_event_copy.notify_all();
                    break;
                }
                
                if let Ok(OwnedMessage::Text(raw_data)) = message {
                    let parsed_command = serde_json::from_str(raw_data.as_str());
                    let command_result = if let Ok(command) = parsed_command {
                        Ok((command, Instant::now()))
                    } else {
                        Err(CommandError::ParseError)
                    };

                    let mut locked_queue = queue_copy.lock().unwrap();
                    locked_queue.push_back(command_result);
                    message_event_copy.notify_all();
                }
            }
        });

        PlayerCommunicator {
            writer,
            queue,
            message_event,
            disconnected,
        }
    }

    fn send(&mut self, message: &Message) {
        if let Err(e) = self.writer.send_message(message) {
            debug!("Send error: {:?}", e);
        }
    }

    fn recv(&self, wait_timeout: u64) -> Result<(Command, Instant), CommandError> {
        // Try to pop message
        if self.disconnected.load(Ordering::Relaxed) {
            return Err(CommandError::Disconnected);
        }
        let mut locked_queue = self.queue.lock().unwrap();
        if let Some(result) = locked_queue.pop_front() {
            return result;
        }
        
        // Wait for command
        let (mut locked_queue, _) = self.message_event.wait_timeout(
            locked_queue, Duration::from_millis(wait_timeout)
        ).unwrap();
        
        // Then try to pop message one more time
        if self.disconnected.load(Ordering::Relaxed) {
            return Err(CommandError::Disconnected);
        }
        if let Some(result) = locked_queue.pop_front() {
            return result;
        }
        
        Err(CommandError::ReadTimeoutError)
    }
}

fn send_all_alive(message: &Message, players: &mut Vec<Option<PlayerCommunicator>>) {
    players
        .iter_mut()
        .filter(|player| player.is_some())
        .map(|player| player.as_mut().unwrap())
        .for_each(|player| {
            player.send(message);
        });
}

pub fn game_loop(
    players: Vec<Client<TcpStream>>,
    game_config: GameConfig,
    wait_timeout: u64,
) {
    let players_count = players.len();
    let mut players: Vec<_> = players
        .into_iter()
        .map(|player| Some(PlayerCommunicator::new(player)))
        .collect();

    let mut ranks = vec![0usize; players_count];
    let mut current_rank = 0;
    
    let mut game;
    {
        let areas;
        (game, areas) = Game::from_config(game_config);

        // TODO: Add command line flag "--polygons" that enables "areas" field in config 
        //  There is no need to get this polygons if you don't visualize game.
        let mut config = json!({
            "areas": get_polygons(areas, 0.5),
            "graph": game.graph_ref(),
            "eliminate_every_n_round": game.get_eliminate_every_n_round(),
            "timeout": wait_timeout
        });
        info!("{}", serde_json::to_string(&config).unwrap());
        
        
        for (player_id, player) in players.iter_mut().enumerate() {
            config["me"] = Value::from(player_id);
            player.as_mut().unwrap().send(&Message::text(json!({"start": config}).to_string()));
        }
    }

    let close_message = &Message::close_because(1000, "Eliminated");
    
    while !game.is_ended() {
        info!("{}", serde_json::to_string(&game).unwrap());
        let state  = &Message::text(json!({"state": game}).to_string());
        send_all_alive(state, &mut players);

        {
            let player_id = game.get_current_player();
            let wrapped_player = &mut players[player_id];
            let (command, command_error) = match wrapped_player {
                Some(player) => {
                    match player.recv(wait_timeout) {
                        Ok((command, _received_at)) => {
                            // TODO: warnings for premature commands
                            (command, None)
                        }
                        Err(CommandError::Disconnected) => {
                            (Command::EndTurn, Some("Disconnected"))
                        }
                        Err(CommandError::ParseError) => {
                            (Command::EndTurn, Some("Parse error"))
                        }
                        Err(CommandError::ReadTimeoutError) => {
                            (Command::EndTurn, Some("Read timeout"))
                        }
                    }
                }
                None => unreachable!("Eliminated player can't turn")
            };
            
            let mut state = json!({
                "player": player_id,
                "command": serde_json::to_value(&command).unwrap()
            });

            let turn_result = game.turn(command);
            if let Some(err_msg) = command_error {
                state["error"] = Value::String(err_msg.to_string());
            } else if let Err(err) = turn_result {
                state["error"] = Value::String(err.reason);
            }

            info!("{}", state.to_string());
        }

        let mut eliminated = vec![true; players_count];
        for player_id in game.get_players() {
            eliminated[player_id] = false;
        }
        let mut somebody_eliminated = false;

        let end_message = &Message::text(json!({"end": game}).to_string());
        
        players = eliminated
            .into_iter()
            .zip(players.into_iter())
            .enumerate()
            .map(|(player_id, (eliminated, player))| {
                if eliminated {
                    if let Some(mut client) = player {
                        client.send(end_message);
                        client.send(close_message);

                        ranks[player_id] = current_rank;
                        somebody_eliminated = true;

                        return None;
                    }
                }

                player
            })
            .collect();
        
        if somebody_eliminated {
            current_rank += 1;
        }
    }
    
    let end_message = &Message::text(json!({"end": game}).to_string());
    send_all_alive(end_message, &mut players);
    send_all_alive(close_message, &mut players);
    
    info!("{}", serde_json::to_string(&game).unwrap());

    // Fill ranks of winner
    for player_id in game.get_players() {
        ranks[player_id] = current_rank;
    }

    // Invert ranks. Less is better.
    for i in 0..players_count {
        ranks[i] = current_rank - ranks[i];
    }
    
    info!("{}", json!({"ranks": ranks}).to_string());
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
