#![feature(hash_drain_filter)]
#![feature(destructuring_assignment)]
mod game;
mod network;
mod cli;

use clap::Clap;
use websocket::sync::Server;

use crate::cli::Opt;
use crate::game::GameConfig;
use rand::{Rng, thread_rng};

fn main() {
    let opt = Opt::parse();

    let seed = opt.seed.unwrap_or_else(|| {
        thread_rng().gen()
    });

    let game_config = GameConfig::new(
        opt.players,
        opt.eliminate_every_n_round,
        opt.areas,
        opt.max_area_size,
        opt.min_area_size,
        opt.spread,
        opt.grow,
        seed,
    );

    let server = Server::bind(opt.address).unwrap();
    let clients = network::connection_handler(server, opt.players);
    
    network::game_loop(
        clients,
        game_config,
        opt.wait_timeout,
    );
}
