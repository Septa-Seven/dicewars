use std::ops::Range;
use std::collections::{HashMap, HashSet};
use std::iter::repeat_with;
use rand::{self, Rng};
use indexmap::IndexSet;


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

    let mut field = HashMap::with_capacity(sizes.iter().sum());
    
    let mut can_expand_from = IndexSet::new();
    can_expand_from.insert((0, 0));

    for area_index in 0..areas_count {
        'retry_area_generation: loop {
            // Pick empty hex to start area from
            let start = *can_expand_from.get_index(random.gen_range(0..can_expand_from.len())).unwrap();
            let mut possible_expantion = IndexSet::new();
            possible_expantion.insert(start);
            
            let mut size = sizes[area_index];
            let mut area = Vec::new();
            
            while size > 0 {
                if possible_expantion.is_empty() {
                    for a in area.iter() {
                        field.remove(a);
                    }
                    continue 'retry_area_generation;
                }

                let &expand = possible_expantion.get_index(random.gen_range(0..possible_expantion.len())).unwrap();
                possible_expantion.remove(&expand);
                
                field.insert(expand, area_index);
                area.push(expand);

                let directions = if expand.1 % 2 == 0 {EVEN_DIRECTIONS} else {ODD_DIRECTIONS};
                for direction in directions.iter() {
                    let neighbor = (expand.0 + direction.0, expand.1 + direction.1);
                    
                    if let Some(&neighbor_area_index) = field.get(&neighbor) {
                        if neighbor_area_index != area_index {
                            graph[neighbor_area_index].insert(area_index);
                            graph[area_index].insert(neighbor_area_index);
                        }
                    } else {
                        possible_expantion.insert(neighbor);
                    }
                }

                size -= 1;
            }

            can_expand_from.extend(possible_expantion);

            for a in area {
                can_expand_from.remove(&a);
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
