use std::collections::{HashMap, HashSet};
use crate::generation::AreaGraph;
use rand::{thread_rng, Rng};
use rand::seq::SliceRandom;
use std::iter::repeat;
use serde::{Serialize, Deserialize};

#[derive(Serialize)]
struct Player {
    id: usize,
    savings: u32,
}

impl Player {
    fn new(id: usize) -> Player {
        Player {
            id: id,
            savings: 0
        }
    }
}

#[derive(Serialize)]
struct Area {
    owner: Option<usize>,
    dices: u32,
}

impl Area {
    fn new(initial_dices: u32, owner: Option<usize>) -> Area {
        Area {
            owner: owner,
            dices: initial_dices,
        }
    }

    fn roll_dices(&self) -> u32 {
        let mut random = thread_rng();
        random.gen_range(self.dices..self.dices * 6 + 1)
    }
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "snake_case", tag = "type", content = "data")]
pub enum Command {
    Attack {from: usize, to: usize},
    EndTurn,
}

pub struct ValidationError {
    pub reason: String,
}

#[derive(Serialize)]
pub struct Game {
    round: u32,
    #[serde(skip)]
    eliminate_every_n_round: u32,
    current_player_index: usize,
    players: Vec<Player>,
    #[serde(skip)]
    graph: AreaGraph,
    areas: Vec<Area>,
}

impl Game {
    pub fn new(players_count: usize, eliminate_every_n_round: u32,
        areas_graph: AreaGraph, spread: f32, grow: f32) -> Game {
        assert!(players_count > 1 && players_count < 9);
        assert!(players_count <= areas_graph.len());
        assert!(0.0 <= grow || grow <= 1.0);
        assert!(0.0 <= spread || spread <= 1.0);

        let random = &mut thread_rng();
        let areas_per_player = (
            (areas_graph.len() as f32 * spread) as usize / players_count
        ).max(players_count);
        
        let players_area_count = areas_per_player * players_count;
        let neutral_area_count = areas_graph.len() - players_area_count;
        let area_pick_player: Vec<Option<usize>> = 
            // Each player take <areas_per_player> areas
            (0..players_count)
            .map(|player_id| repeat(player_id).take(areas_per_player))
            .flatten()
            .map(|player| Some(player))
            // Remaining areas are neutral
            .chain(repeat(None).take(neutral_area_count))
            .collect();
        
        let areas = area_pick_player
            .into_iter()
            .map(|player| Area::new(1, player))
            .collect();
        
        let players = (0..players_count)
            .map(|player_id| Player::new(player_id))
            .collect();
        
        let mut game = Game {
            round: 0,
            current_player_index: 0,
            players: players,
            areas: areas,
            graph: areas_graph,
            eliminate_every_n_round: eliminate_every_n_round,
        };

        // Grow players
        let additional_player_dices = ((areas_per_player * 7) as f32 * grow) as u32;
        for player_id in 0..players_count {
            let player_areas = ((player_id * areas_per_player)..((player_id + 1) * areas_per_player)).collect();
            game.grow_areas(player_areas, additional_player_dices);
        }

        // Grow neutrals proportionaly to players
        let additional_neutral_dices = (neutral_area_count * additional_player_dices as usize / areas_per_player) as u32;
        let neutral_areas = (game.areas.len() - neutral_area_count..game.areas.len()).collect();
        game.grow_areas(neutral_areas, additional_neutral_dices);

        game.areas.shuffle(random);

        game
    }

    pub fn turn(&mut self, command: Command) -> Result<(), ValidationError> {
        let mut ret = Ok(());
        if let Command::Attack {from, to} = command {
            ret = self.attack(from, to);
            if let Ok(_) = ret {
                return Ok(());
            }
        }

        self.end_turn();

        ret
    }

    fn attack(&mut self, from: usize, to: usize) -> Result<(), ValidationError> {
        if from == to {
            return Err(ValidationError {reason: String::from("Same areas.")});
        }
        
        if from >= self.graph.len() {
            return Err(ValidationError {reason: String::from("There is no such area to start attack from.")});
        }
        
        if !self.graph[from].contains(&to) {
            return Err(ValidationError {reason: String::from("There is no such traversal.")});
        }
        
        let to_owner;
        {
            let player_id = self.players[self.current_player_index].id;
            let from_area = &self.areas[from];
            if from_area.dices == 1 {
                return Err(ValidationError {reason: String::from("To attack from an area it must have more then one die.")});
            }
    
            if from_area.owner.is_none() || from_area.owner.unwrap() != player_id {
                return Err(ValidationError {reason: String::from("This area is not yours.")});
            }

            let to_area = &self.areas[to];
            to_owner = to_area.owner;
            if let Some(not_neutral) = to_owner {
                if not_neutral == player_id {
                    return Err(ValidationError {reason: String::from("You can't attack your own area.")});
                }
            }
        }

        // Fight
        let from_roll = self.areas[from].roll_dices();
        let to_roll = self.areas[to].roll_dices();

        if from_roll > to_roll {
            self.areas[to].dices = self.areas[from].dices - 1;
            self.areas[to].owner = self.areas[from].owner;
            
            // Offended player with no areas left will be eliminated immediately
            if to_owner.is_some() {
                let areas_count = self.areas
                    .iter()
                    .filter(|area| area.owner == to_owner)
                    .count();
                
                if areas_count == 0 {
                    let offended_player = to_owner.unwrap();
                    let offended_index = self.players
                        .iter()
                        .position(|player| player.id == offended_player)
                        .unwrap();
                    
                    self.players.remove(offended_index);
                }
            }
        }
        self.areas[from].dices = 1;

        Ok(())
    }

    fn end_turn(&mut self) {
        let player_id = self.players[self.current_player_index].id;
        let mut not_full_areas = Vec::new();
        
        // Max income from connected areas of current player
        let mut income = {
            let mut incomes = Vec::new();
            let mut visited = vec![false; self.areas.len()];
            
            for area_index in 0..self.areas.len() {
                if visited[area_index] {
                    continue;
                }
                visited[area_index] = true;

                {
                    let owner = self.areas[area_index].owner;
                    if owner.is_none() || owner.unwrap() != player_id {
                        continue;
                    }
                }

                let mut income: u32 = 0;
                let mut to_check = vec![area_index];
                
                
                while let Some(area_index) = to_check.pop() {
                    let area = &self.areas[area_index];
                    
                    let owner = self.areas[area_index].owner;
                    if owner.is_none() || owner.unwrap() != player_id {
                        continue;
                    }

                    if area.dices < 8 {
                        not_full_areas.push(area_index);
                    }
                    income += 1;

                    let neighbors = &self.graph[area_index];
                    for &neighbor_index in neighbors {
                        if visited[neighbor_index] {
                            continue;
                        }
                        visited[neighbor_index] = true;
                        to_check.push(neighbor_index);
                    }
                }
    
                incomes.push(income);
            }

            *incomes.iter().max().unwrap()
        };

        // Spend savings
        let player = &self.players[self.current_player_index];
        income += player.savings;
        income = self.grow_areas(not_full_areas, income);

        let player = &mut self.players[self.current_player_index];
        player.savings = income;

        // Next turn
        self.current_player_index += 1;
        
        // Elimination
        let mut area_counts: HashMap<usize, u32> = self.players
            .iter()
            .map(|player| (player.id, 0))
            .collect();
        
        for area in self.areas.iter() {
            if let Some(owner) = area.owner {
                *area_counts.get_mut(&owner).unwrap() += 1;
            }
        }

        let mut remove_players: HashSet<usize> = area_counts
            .drain_filter(|_, &mut area_count| area_count == 0)
            .map(|(player_id, _)| player_id)
            .collect();
        
        if self.current_player_index == self.players.len() {
            self.round += 1;
            self.current_player_index = 0;

            if self.round % self.eliminate_every_n_round == 0 {
                let min_area_count = *area_counts.values().min().unwrap();
                let eliminated_players: HashSet<_> = area_counts
                    .drain_filter(|_, &mut area_count| area_count == min_area_count)
                    .map(|(player_id, _)| player_id)
                    .collect();

                
                for area in &mut self.areas {
                    if let Some(owner) = area.owner {
                        if eliminated_players.contains(&owner) {
                            area.owner = None;
                        }
                    }
                }

                remove_players.extend(eliminated_players);
            }
        }
        
        self.players.retain(|player| !remove_players.contains(&player.id));
    }

    fn grow_areas(&mut self, mut area_indices: Vec<usize>, mut count: u32) -> u32 {
        let random = &mut thread_rng();
        while count > 0 && !area_indices.is_empty() {
            let i = random.gen_range(0..area_indices.len());
            let area = &mut self.areas[area_indices[i]];
            
            // Can't grow area beyond maximum size
            if area.dices == 8 {
                area_indices.remove(i);
                continue;
            }
            
            // Grow area
            area.dices += 1;
            count -= 1;
        }

        count
    }

    pub fn is_ended(&self) -> bool {
        self.players.len() < 2
    }

    pub fn get_current_player(&self) -> usize {
        self.players[self.current_player_index].id
    }

    pub fn get_players(&self) -> Vec<usize> {
        self.players.iter().map(|player| player.id).collect()
    }
}
