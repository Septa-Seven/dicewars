use std::net::SocketAddr;
use clap::{Clap};


// TODO: description
#[derive(Clap)]
#[clap(name = "dicewars")]
pub struct Opt {
    #[clap(long, default_value = "127.0.0.1:8000")]
    pub address: SocketAddr,

    #[clap(short, long)]
    pub players: usize,

    #[clap(long)]
    pub area_count: usize,

    #[clap(long)]
    pub eliminate_every_n_round: u32,

    #[clap(long)]
    pub min_area_size: usize,

    #[clap(long)]
    pub max_area_size: usize,

    // TODO: validate 0.0 <= x <= 1.0
    #[clap(long)]
    pub spread: f32,

    #[clap(long)]
    pub grow: f32,

    #[clap(short, long)]
    pub verbose: bool,
}