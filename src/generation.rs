use std::ops::Range;
use std::collections::{HashMap, HashSet};
use std::iter::repeat_with;
use rand::{self, Rng};
use indexmap::{IndexMap, IndexSet};


type Point = (i32, i32);
const EVEN_DIRECTIONS: [Point; 6] = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, 0), (0, 1)];
const ODD_DIRECTIONS: [Point; 6] = [(0, 1), (1, 0), (0, -1), (-1, -1), (-1, 0), (-1, 1), ];

const BORDERS: [((f32, f32), (f32, f32)); 6] = [
    ((0.0, 0.5), (0.5, 0.25)),
    ((0.5, 0.25), (0.5, -0.25)),
    ((0.5, -0.25), (0.0, -0.5)),
    ((0.0, -0.5), (-0.5, -0.25)),
    ((-0.5, -0.25), (-0.5, 0.25)),
    ((-0.5, 0.25), (0.0, 0.5)),
];

pub type AreaGraph = Vec<HashSet<usize>>;

pub fn generate_areas(areas_count: usize, area_size_range: Range<usize>) -> (Vec<HashSet<Point>>, AreaGraph) {
    let random = &mut rand::thread_rng();

    let sizes: Vec<usize> = repeat_with(|| random.gen_range(area_size_range.clone()))
        .take(areas_count)
        .collect();
    
    let mut graph: Vec<HashSet<usize>> = repeat_with(HashSet::new)
        .take(areas_count)
        .collect();
    
    let mut possible_start = IndexSet::new();
    possible_start.insert((0, 0));

    let mut field = HashMap::with_capacity(sizes.iter().sum());
    
    for area_index in 0..areas_count {
        'retry_area_generation: loop {
            // Pick empty hex to start area from
            let start_hex_index= random.gen_range(0..possible_start.len());
            let &start_hex = possible_start.get_index(start_hex_index).unwrap();


            let mut neighbors_count = IndexMap::new();
            let mut max_neighbors_count = 0;
            neighbors_count.insert((0, 0), 0);
        
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
                                let group = &mut count_groups[*count];
                                group.remove(&neighbor);
                            }
                            *count += 1;
                            *count
                        } else {
                            neighbors_count.insert(neighbor, 0);
                            0
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

    let mut areas = vec![HashSet::new(); areas_count];

    for (position, area) in field {
        areas[area].insert(position);
    }

    (areas, graph)
}

pub fn get_polygons(areas: Vec<HashSet<Point>>) -> Vec<Vec<(f32, f32)>> {
    
    areas
        .into_iter()
        .map(|area| {
            let mut polygon = Vec::new();
            
            for hex in area.iter() {
                let hex_real_coords = (hex.0 as f32 - 0.5 * (hex.1 % 2 != 0) as u32 as f32, hex.1 as f32 * 0.75);

                let directions = if hex.1 % 2 == 0 {EVEN_DIRECTIONS} else {ODD_DIRECTIONS};
                for (direction, border_direction) in directions.iter().zip(BORDERS.iter()) {
                    let neighbor = (hex.0 + direction.0, hex.1 + direction.1);

                    if !area.contains(&neighbor) {
                        let border = (
                            (hex_real_coords.0 + border_direction.0.0, hex_real_coords.1 + border_direction.0.1),
                            (hex_real_coords.0 + border_direction.1.0, hex_real_coords.1 + border_direction.1.1),
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
