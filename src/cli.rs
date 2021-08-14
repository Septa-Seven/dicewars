use std::net::SocketAddr;
use clap::{Clap};


#[derive(Clap)]
#[clap(name = "dicewars")]
pub struct Opt {
    #[clap(long, default_value = "127.0.0.1:8000")]
    pub address: SocketAddr,

    #[clap(short, long)]
    pub players: usize,

    #[clap(short, long)]
    pub min_: bool,

    #[clap(short, long)]
    pub verbose: bool

}