#![feature(hash_drain_filter)]
#![feature(destructuring_assignment)]
mod game;
mod network;
mod cli;

use clap::Clap;
use log::{LevelFilter};
use log4rs::append::console::ConsoleAppender;
use log4rs::config::{Appender, Config, Root};
use log4rs::encode::pattern::PatternEncoder;
use websocket::sync::Server;

use crate::cli::Opt;
use crate::game::GameConfig;

fn init_logging(verbose: bool) {
    let log_level = if verbose {LevelFilter::Info} else {LevelFilter::Error};
    
    let stdout = ConsoleAppender::builder()
        .encoder(Box::new(PatternEncoder::new("{m}\n")))
        .build();
    let config = Config::builder()
        .appender(Appender::builder().build("stdout", Box::new(stdout)))
        .build(Root::builder().appender("stdout").build(log_level))
        .unwrap();

    let _handle = log4rs::init_config(config).unwrap();
}

fn main() {
    let opt = Opt::parse();

    let game_config = GameConfig::new(
        opt.players,
        opt.eliminate_every_n_round,
        opt.areas,
        opt.max_area_size,
        opt.min_area_size,
        opt.spread,
        opt.grow,
    );

    init_logging(opt.verbose);
    let server = Server::bind(opt.address).unwrap();
    let clients = network::connection_handler(server, opt.players);
    
    network::game_loop(
        clients,
        game_config,
        opt.wait_timeout,
    );
}
