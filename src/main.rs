#![feature(hash_drain_filter)]
mod generation;
mod game;
mod network;

use crate::network::connection_handler;

fn main() {
    connection_handler();
}
