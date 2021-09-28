use indexmap::IndexSet;
use std::collections::{HashMap, HashSet};
use rand::{thread_rng, Rng};
use rand::seq::SliceRandom;
use std::iter::{repeat, repeat_with};
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

pub struct GameConfig {
    players: usize,
    areas: usize,
    eliminate_every_n_round: u32,
    max_area_size: usize,
    min_area_size: usize,
    spread: f32,
    grow: f32,
}

impl GameConfig {
    pub fn new(players: usize, eliminate_every_n_round: u32, areas: usize,
        max_area_size: usize, min_area_size: usize, spread: f32, grow: f32) -> GameConfig {
        
        assert!(players > 1 && players < 9);
        assert!(players <= areas);
        assert!(0.0 <= grow && grow <= 1.0);
        assert!(0.0 <= spread && spread <= 1.0);
        assert!(min_area_size > 0);
        assert!(max_area_size > 0);
        assert!(eliminate_every_n_round > 0);
        assert!(min_area_size < max_area_size);

        GameConfig {
            players,
            areas,
            eliminate_every_n_round,
            max_area_size,
            min_area_size,
            spread,
            grow,
        }
    }
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
    pub fn from_config(game_config: GameConfig) -> (Game, Areas) {
        let (areas_coords, graph) = generate_areas(&game_config);

        let random = &mut thread_rng();
        let areas_per_player = (
            (graph.len() as f32 * game_config.spread) as usize / game_config.players
        ).max(game_config.players);
        
        let players_area_count = areas_per_player * game_config.players;
        let neutral_area_count = graph.len() - players_area_count;
        let area_pick_player: Vec<Option<usize>> = 
            // Each player take <areas_per_player> areas
            (0..game_config.players)
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
        
        let players = (0..game_config.players)
            .map(|player_id| Player::new(player_id))
            .collect();
        
        let mut game = Game {
            round: 0,
            current_player_index: 0,
            players: players,
            areas: areas,
            graph: graph,
            eliminate_every_n_round: game_config.eliminate_every_n_round,
        };

        // Grow players
        let additional_player_dices = ((areas_per_player * 7) as f32 * game_config.grow) as u32;
        for player_id in 0..game_config.players {
            let player_areas = ((player_id * areas_per_player)..((player_id + 1) * areas_per_player)).collect();
            game.grow_areas(player_areas, additional_player_dices);
        }

        // Grow neutrals proportionaly to players
        let additional_neutral_dices = (neutral_area_count * additional_player_dices as usize / areas_per_player) as u32;
        let neutral_areas = (game.areas.len() - neutral_area_count..game.areas.len()).collect();
        game.grow_areas(neutral_areas, additional_neutral_dices);

        game.areas.shuffle(random);

        (game, areas_coords)
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

                    if offended_index < self.current_player_index {
                        self.current_player_index -= 1;
                    }
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
        let mut dice_counts: HashMap<usize, u32> = self.players
            .iter()
            .map(|player| (player.id, player.savings))
            .collect();
        
        for area in self.areas.iter() {
            if let Some(owner) = area.owner {
                *dice_counts.get_mut(&owner).unwrap() += area.dices;
            }
        }

        let mut remove_players: HashSet<usize> = dice_counts
            .drain_filter(|_, &mut dice_count| dice_count == 0)
            .map(|(player_id, _)| player_id)
            .collect();
        
        if self.current_player_index == self.players.len() {
            self.round += 1;
            self.current_player_index = 0;

            if self.round % self.eliminate_every_n_round == 0 {
                let min_dice_count = *dice_counts.values().min().unwrap();
                let eliminated_players: HashSet<_> = dice_counts
                    .drain_filter(|_, &mut dice_count| dice_count == min_dice_count)
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

    pub fn graph_ref(&self) -> &AreaGraph {
        &self.graph
    }

    pub fn get_eliminate_every_n_round(&self) -> u32 {
        self.eliminate_every_n_round
    }
}

type Point = (i32, i32);
const EVEN_DIRECTIONS: [Point; 6] = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, 0), (0, 1)];
const ODD_DIRECTIONS: [Point; 6] = [(0, 1), (1, 0), (0, -1), (-1, -1), (-1, 0), (-1, 1)];

// (60.0 * PI / 180.0).cos();
const COS: f32 = 0.49999787927;


pub type AreaGraph = Vec<HashSet<usize>>;
pub type Areas = Vec<HashSet<Point>>;

fn generate_areas(game_config: &GameConfig) -> (Areas, AreaGraph) {
    let random = &mut rand::thread_rng();

    let area_size_range = game_config.min_area_size..game_config.max_area_size + 1;
    let sizes: Vec<usize> = repeat_with(|| random.gen_range(area_size_range.clone()))
        .take(game_config.areas)
        .collect();
    
    let mut graph: Vec<HashSet<usize>> = repeat_with(HashSet::new)
        .take(game_config.areas)
        .collect();
    
    let mut possible_start = IndexSet::new();
    possible_start.insert((0, 0));

    let mut field = HashMap::with_capacity(sizes.iter().sum());
    
    for area_index in 0..game_config.areas {
        'retry_area_generation: loop {
            // Pick empty hex to start area from
            let start_hex_index = random.gen_range(0..possible_start.len());
            let &start_hex = possible_start.get_index(start_hex_index).unwrap();

            let mut neighbors_count = HashMap::new();
            let mut max_neighbors_count = 0;
        
            let mut count_groups = vec![IndexSet::new(); 5];
            count_groups[max_neighbors_count].insert(start_hex);
            
            let mut size = sizes[area_index];
            let mut area = Vec::new();
            
            while size > 0 {
                let expand_hex;
                {
                    let group = loop {
                        let group = &mut count_groups[max_neighbors_count];
                        if !group.is_empty() {
                            break group;
                        }
                        else if max_neighbors_count == 0 {
                            for hex in area.iter() {
                                field.remove(hex);
                                let neighbors: Vec<_> = graph[area_index].iter().copied().collect();
                                for neigbor_index in neighbors {
                                    graph[neigbor_index].remove(&area_index);
                                }
                                graph[area_index].clear();
                            }
                            continue 'retry_area_generation;
                        }
                        else {
                            max_neighbors_count -= 1;
                        }
                    };
                    
                    let expand_hex_index = random.gen_range(0..group.len());
                    expand_hex = *group.get_index(expand_hex_index).unwrap();
                    group.remove(&expand_hex);
                }

                field.insert(expand_hex, area_index);
                area.push(expand_hex);

                let directions = if expand_hex.1 % 2 == 0 {EVEN_DIRECTIONS} else {ODD_DIRECTIONS};
                for direction in directions.iter() {
                    let neighbor = (expand_hex.0 + direction.0, expand_hex.1 + direction.1);
                    
                    if let Some(&neighbor_area_index) = field.get(&neighbor) {
                        if neighbor_area_index != area_index {
                            graph[neighbor_area_index].insert(area_index);
                            graph[area_index].insert(neighbor_area_index);
                        }
                    } else {
                        let count = if let Some(count) = neighbors_count.get_mut(&neighbor) {
                            {
                                let group: &mut IndexSet<_> = &mut count_groups[*count];
                                group.remove(&neighbor);
                            }
                            *count += 1;
                            *count
                        } else {
                            neighbors_count.insert(neighbor, 1);
                            1
                        };

                        if count != 6 {
                            let group = &mut count_groups[count];
                            group.insert(neighbor);
                        
                            if max_neighbors_count < count {
                                max_neighbors_count = count;
                            }
                        }
                    }
                }

                size -= 1;
            }
            
            possible_start.extend(neighbors_count.keys());

            for a in area.iter() {
                possible_start.remove(a);
            }
            break;
        }
    }

    let mut areas = vec![HashSet::new(); game_config.areas];

    for (position, area) in field {
        areas[area].insert(position);
    }

    (areas, graph)
}

pub fn get_polygons(areas: Areas, width: f32) -> Vec<Vec<(f32, f32)>> {
    // Width of hex side is 0.5
    // TODO: Generate borders based on any width
    let width_half = width / 2.0;
    let r = (
        width * width + width_half * width_half - width * width * COS
    ).sqrt();
    
    let borders = [
        ((0.0, width), (r, width_half)),
        ((r, width_half), (r, -width_half)),
        ((r, -width_half), (0.0, -width)),
        ((0.0, -width), (-r, -width_half)),
        ((-r, -width_half), (-r, width_half)),
        ((-r, width_half), (0.0, width)),
    ];
    let y_shift = 1.5 * width;

    areas
        .into_iter()
        .map(|area| {
            let mut polygon = Vec::new();
            
            for hex in area.iter() {
                let hex_real_coords = (
                    r * (2.0 * hex.0 as f32 - (hex.1 % 2 != 0) as u32 as f32),
                    hex.1 as f32 * y_shift
                );
                
                let directions = if hex.1 % 2 == 0 {EVEN_DIRECTIONS} else {ODD_DIRECTIONS};
                for (direction, border_direction) in directions.iter().zip(borders.iter()) {
                    let neighbor = (hex.0 + direction.0, hex.1 + direction.1);

                    if !area.contains(&neighbor) {
                        let border = (
                            (
                                ((hex_real_coords.0 + border_direction.0.0) * 1000.0).round() / 1000.0,
                                ((hex_real_coords.1 + border_direction.0.1) * 1000.0).round() / 1000.0
                            ),
                            (
                                ((hex_real_coords.0 + border_direction.1.0) * 1000.0).round() / 1000.0,
                                ((hex_real_coords.1 + border_direction.1.1) * 1000.0).round() / 1000.0,
                            )
                        );

                        polygon.push(border);
                    }
                }
            }

            // Sort polygon edges
            for i in 0..polygon.len() - 2 {
                let edge = polygon[i];
                let start_check = i + 1;
                
                for j in start_check..polygon.len() {
                    let check = polygon[j];

                    if edge.1 == check.0 {
                        polygon.swap(start_check, j);
                        break;
                    }
                }
            }
            
            polygon
                .iter()
                .map(|edge| edge.0)
                .collect()
        })
    .collect()
}
