use std::time::Instant;
use std::time::{SystemTime, UNIX_EPOCH};

const NUM_ITEMS: i32 = 20; // A reasonable value for exhaustive search.
                           //const NUM_ITEMS: i32 = 35;    // A reasonable value for branch and bound.

const MIN_VALUE: i32 = 1;
const MAX_VALUE: i32 = 10;
const MIN_WEIGHT: i32 = 4;
const MAX_WEIGHT: i32 = 10;

struct Item {
    id: i32,
    value: i32,
    weight: i32,
    is_selected: bool,
    blocked_by: i32,
    block_list: Vec<i32>,
}

// ************
// *** Prng ***
// ************
struct Prng {
    seed: u32,
}

impl Prng {
    fn new() -> Self {
        let mut prng = Self { seed: 0 };
        prng.randomize();
        return prng;
    }

    fn randomize(&mut self) {
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_millis();
        self.seed = millis as u32;
    }

    // Return a pseudorandom value in the range [0, 2147483647].
    fn next_u32(&mut self) -> u32 {
        self.seed = self.seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        self.seed %= 1 << 31;
        return self.seed;
    }

    // Return a pseudorandom value in the range [0.0, 1.0).
    fn next_f64(&mut self) -> f64 {
        let f = self.next_u32() as f64;
        return f / (2147483647.0 + 1.0);
    }

    // Return a pseudorandom value in the range [min, max).
    fn next_i32(&mut self, min: i32, max: i32) -> i32 {
        let range = (max - min) as f64;
        let result = min as f64 + range * self.next_f64();
        return result as i32;
    }
}

// Make some random items.
fn make_items(
    prng: &mut Prng,
    num_items: i32,
    min_value: i32,
    max_value: i32,
    min_weight: i32,
    max_weight: i32,
) -> Vec<Item> {
    let mut items: Vec<Item> = Vec::with_capacity(num_items as usize);
    for i in 0..num_items {
        let item = Item {
            id: i,
            value: prng.next_i32(min_value, max_value),
            weight: prng.next_i32(min_weight, max_weight),
            is_selected: false,
            blocked_by: -1,
            block_list: Vec::new(),
        };
        items.push(item);
    }
    return items;
}

// Return a copy of the items.
fn copy_items(items: &mut Vec<Item>) -> Vec<Item> {
    let mut new_items: Vec<Item> = Vec::with_capacity(items.len());
    for item in items {
        let new_item = Item {
            id: item.id,
            value: item.value,
            weight: item.weight,
            is_selected: item.is_selected,
            blocked_by: item.blocked_by,
            block_list: item.block_list.clone(),
        };
        new_items.push(new_item);
    }
    return new_items;
}

// Return the total value of the items.
// If add_all is false, only add up the selected items.
fn sum_values(items: &mut Vec<Item>, add_all: bool) -> i32 {
    let mut total = 0;
    for i in 0..items.len() {
        if add_all || items[i].is_selected {
            total += items[i].value;
        }
    }
    return total;
}

// Return the total weight of the items.
// If add_all is false, only add up the selected items.
fn sum_weights(items: &mut Vec<Item>, add_all: bool) -> i32 {
    let mut total = 0;
    for i in 0..items.len() {
        if add_all || items[i].is_selected {
            total += items[i].weight;
        }
    }
    return total;
}

// Return the value of this solution.
// If the solution is too heavy, return -1 so we prefer an empty solution.
fn solution_value(items: &mut Vec<Item>, allowed_weight: i32) -> i32 {
    // If the solution's total weight > allowed_weight,
    // return 0 so we won't use this solution.
    if sum_weights(items, false) > allowed_weight {
        return -1;
    }

    // Return the sum of the selected values.
    return sum_values(items, false);
}

// Print the selected items.
fn print_selected(items: &mut Vec<Item>) {
    let mut num_printed = 0;
    for i in 0..items.len() {
        if items[i].is_selected {
            print!("{}({}, {}) ", i, items[i].value, items[i].weight)
        }
        num_printed += 1;
        if num_printed > 100 {
            println!("...");
            return;
        }
    }
    println!();
}

// Run the algorithm. Display the elapsed time and solution.
fn run_algorithm(
    alg: &dyn Fn(&mut Vec<Item>, i32) -> (Vec<Item>, i32, i32),
    items: &mut Vec<Item>,
    allowed_weight: i32,
) {
    // Copy the items so the run isn't influenced by a previous run.
    let mut test_items = copy_items(items);

    let start = Instant::now();

    // Run the algorithm.
    let mut solution: Vec<Item>;
    let total_value: i32;
    let function_calls: i32;
    (solution, total_value, function_calls) = alg(&mut test_items, allowed_weight);

    let duration = start.elapsed();
    println!("Elapsed: {:?}", duration);

    print_selected(&mut solution);
    println!(
        "Value: {}, Weight: {}, Calls: {}",
        total_value,
        sum_weights(&mut solution, false),
        function_calls
    );
    println!();
}

// Recursively assign values in or out of the solution.
// Return the best assignment, value of that assignment,
// and the number of function calls we made.
fn exhaustive_search(items: &mut Vec<Item>, allowed_weight: i32) -> (Vec<Item>, i32, i32) {
    return do_exhaustive_search(items, allowed_weight, 0);
}

fn do_exhaustive_search(
    items: &mut Vec<Item>,
    allowed_weight: i32,
    next_index: i32,
) -> (Vec<Item>, i32, i32) {
    let unext = next_index as usize;
    if unext == items.len() {
        // Reached the bottom (i.e. a leaf/terminal node) of the decision tree
        let items_copy = copy_items(items);
        let sol_val = solution_value(items, allowed_weight);
        return (items_copy, sol_val, 1);
    } else {
        // Processing an internal node of the decision tree
        // Perform exhaustive search of the left subtree
        {
            let item: &mut Item = items.get_mut(unext).unwrap();
            item.is_selected = true;
        }
        let (left_subtree_result, left_subtree_value, left_fn_calls) =
            do_exhaustive_search(items, allowed_weight, next_index + 1);

        // Perform exhaustive search of the right subtree
        {
            let item: &mut Item = items.get_mut(unext).unwrap();
            item.is_selected = false;
        }
        let (right_subtree_result, right_subtree_value, right_fn_calls) =
            do_exhaustive_search(items, allowed_weight, next_index + 1);

        // Calculate total function calls
        let total_fn_calls = left_fn_calls + right_fn_calls + 1;

        // Compare the values of the left and right subtrees and return the better one
        if left_subtree_value > right_subtree_value {
            return (left_subtree_result, left_subtree_value, total_fn_calls);
        } else {
            return (right_subtree_result, right_subtree_value, total_fn_calls);
        }
    }
}

fn branch_and_bound(items: &mut Vec<Item>, allowed_weight: i32) -> (Vec<Item>, i32, i32) {
    let mut best_value = 0;
    let (current_value, current_weight, remaining_value) = (0, 0, sum_values(items, true));
    return do_branch_and_bound(
        items,
        allowed_weight,
        0,
        &mut best_value,
        current_value,
        current_weight,
        remaining_value,
    );
}

fn do_branch_and_bound(
    items: &mut Vec<Item>,
    allowed_weight: i32,
    next_index: i32,
    best_value: &mut i32,
    current_value: i32,
    current_weight: i32,
    remaining_value: i32,
) -> (Vec<Item>, i32, i32) {
    let unext = next_index as usize;
    if unext == items.len() {
        // Reached full assignment
        let items_copy = copy_items(items);
        return (items_copy, current_value, 1);
    } else {
        // Check value bound
        if current_value + remaining_value > (*best_value) {
            let mut inc_items = vec![];
            let mut inc_result_value = 0;
            let mut inc_fn_calls = 0;
            // Try including the current item
            if current_weight + items[unext].weight <= allowed_weight {
                items[unext].is_selected = true;

                (inc_items, inc_result_value, inc_fn_calls) = do_branch_and_bound(
                    items,
                    allowed_weight,
                    next_index + 1,
                    best_value,
                    current_value + items[unext].value,
                    current_weight + items[unext].weight,
                    remaining_value - items[unext].value,
                );

                if inc_result_value > *best_value {
                    *best_value = inc_result_value;
                }
            }

            // Try excluding the current item
            let mut exc_items = vec![];
            let mut exc_result_value = 0;
            let mut exc_fn_calls = 0;

            items[unext].is_selected = false;

            if current_value + remaining_value - items[unext].value > *best_value {
                (exc_items, exc_result_value, exc_fn_calls) = do_branch_and_bound(
                    items,
                    allowed_weight,
                    next_index + 1,
                    best_value,
                    current_value,
                    current_weight,
                    remaining_value - items[unext].value,
                );

                if exc_result_value > *best_value {
                    *best_value = exc_result_value;
                }
            }

            let total_fn_calls = inc_fn_calls + exc_fn_calls + 1;

            if inc_result_value > exc_result_value {
                return (inc_items, inc_result_value, total_fn_calls);
            } else {
                return (exc_items, exc_result_value, total_fn_calls);
            }
        } else {
            return (Vec::new(), 0, 1);
        }
    }
}

// Build the items' block lists.
fn make_block_lists(items: &mut Vec<Item>) {
    for i in 0..items.len() {
        if !items[i].block_list.is_empty() {
            items[i].block_list = Vec::new();
        }
        for j in 0..items.len() {
            if i != j && items[i].value >= items[j].value && items[i].weight <= items[j].weight {
                items[i].block_list.push(j.try_into().unwrap());
            }
        }
    }
}

fn rods_technique(items: &mut Vec<Item>, allowed_weight: i32) -> (Vec<Item>, i32, i32) {
    let mut best_value = 0;
    let (current_value, current_weight, remaining_value) = (0, 0, sum_values(items, true));

    make_block_lists(items);
    return do_rods_technique(
        items,
        allowed_weight,
        0,
        &mut best_value,
        current_value,
        current_weight,
        remaining_value,
    );
}

fn do_rods_technique(
    items: &mut Vec<Item>,
    allowed_weight: i32,
    next_index: i32,
    best_value: &mut i32,
    current_value: i32,
    current_weight: i32,
    remaining_value: i32,
) -> (Vec<Item>, i32, i32) {
    let unext = next_index as usize;
    if unext == items.len() {
        // Reached full assignment
        let items_copy = copy_items(items);
        return (items_copy, current_value, 1);
    } else {
        // Check value bound
        if current_value + remaining_value > (*best_value) {
            let mut inc_items = vec![];
            let mut inc_result_value = 0;
            let mut inc_fn_calls = 0;

            /*let mut test1_solution = Vec::new();
            let mut test1_value = 0;
            let mut test1_calls = 1;*/

            // Try including the current item
            if items[unext].blocked_by == -1
                && current_weight + items[unext].weight <= allowed_weight
            {
                items[unext].is_selected = true;

                (inc_items, inc_result_value, inc_fn_calls) = do_rods_technique(
                    items,
                    allowed_weight,
                    next_index + 1,
                    best_value,
                    current_value + items[unext].value,
                    current_weight + items[unext].weight,
                    remaining_value - items[unext].value,
                );

                if inc_result_value > *best_value {
                    *best_value = inc_result_value;
                }
            }

            // Try excluding the current item
            let mut exc_items = vec![];
            let mut exc_result_value = 0;
            let mut exc_fn_calls = 0;

            for item_idx in items[unext].block_list.clone() {
                if items[item_idx as usize].blocked_by == -1 {
                    items[item_idx as usize].blocked_by = next_index;
                }
            }

            items[unext].is_selected = false;

            if current_value + remaining_value - items[unext].value > *best_value {
                (exc_items, exc_result_value, exc_fn_calls) = do_rods_technique(
                    items,
                    allowed_weight,
                    next_index + 1,
                    best_value,
                    current_value,
                    current_weight,
                    remaining_value - items[unext].value,
                );

                if exc_result_value > *best_value {
                    *best_value = exc_result_value;
                }
            }

            for item_idx in items[unext].block_list.clone() {
                if items[item_idx as usize].blocked_by == next_index {
                    items[item_idx as usize].blocked_by = -1;
                }
            }

            let total_fn_calls = inc_fn_calls + exc_fn_calls + 1;

            if inc_result_value > exc_result_value {
                return (inc_items, inc_result_value, total_fn_calls);
            } else {
                return (exc_items, exc_result_value, total_fn_calls);
            }
        } else {
            return (Vec::new(), 0, 1);
        }
    }
}

fn rods_technique_sorted(items: &mut Vec<Item>, allowed_weight: i32) -> (Vec<Item>, i32, i32) {
    let mut best_value = 0;
    let (current_value, current_weight, remaining_value) = (0, 0, sum_values(items, true));

    make_block_lists(items);
    items.sort_by(|a, b| b.block_list.len().cmp(&a.block_list.len()));

    for i in 0..NUM_ITEMS {
        items[i as usize].id = i;
    }

    make_block_lists(items);

    return do_rods_technique(
        items,
        allowed_weight,
        0,
        &mut best_value,
        current_value,
        current_weight,
        remaining_value,
    );
}

// Use Dynamic Programming to find a solution
fn dynamic_programming(items: &mut Vec<Item>, allowed_weight: i32) -> (Vec<Item>, i32, i32) {
    let mut solution_value = vec![vec![0; allowed_weight as usize + 1]; items.len()];
    let mut prev_weight = vec![vec![0; allowed_weight as usize + 1]; items.len()];

    for i in 0..=allowed_weight as usize {
        // Item fits in
        if items[0].weight <= i.try_into().unwrap() {
            solution_value[0][i] = items[0].value;
            prev_weight[0][i] = -1;
        } else {
            prev_weight[0][i] = i.try_into().unwrap();
        }
    }

    for i in 1..NUM_ITEMS as usize {
        for w in 0..=allowed_weight as usize {
            // Item not included
            let exc_value = solution_value[i - 1][w];
            let remaining_weight = w as i32 - items[i].weight;
            // Item included
            if items[i].weight <= w.try_into().unwrap() {
                let prev_best_value = solution_value[i - 1][remaining_weight as usize];
                let curr_best_value = prev_best_value + items[i].value;

                // Compare the values when item included vs item excluded
                if curr_best_value > exc_value {
                    solution_value[i][w] = curr_best_value;
                    prev_weight[i][w] = remaining_weight;
                } else {
                    solution_value[i][w] = exc_value;
                    prev_weight[i][w] = w.try_into().unwrap();
                }
            } else {
                solution_value[i][w] = exc_value;
                prev_weight[i][w] = w.try_into().unwrap();
            }
        }
    }

    for i in 0..NUM_ITEMS as usize {
        items[i].is_selected = false;
    }

    let mut back_i = NUM_ITEMS - 1;
    let mut back_w = allowed_weight;

    while back_i >= 0 {
        if back_w == prev_weight[back_i as usize][back_w as usize] {
            items[back_i as usize].is_selected = false;
        } else {
            items[back_i as usize].is_selected = true;
            back_w = prev_weight[back_i as usize][back_w as usize];
        }
        back_i = back_i - 1;
    }

    return (
        copy_items(items),
        solution_value[NUM_ITEMS as usize - 1][allowed_weight as usize],
        1,
    );
}

fn main() {
    // Prepare a Prng using the same seed each time.
    let mut prng = Prng { seed: 1337 };
    //prng.randomize();

    // Make some random items.
    let mut items = make_items(
        &mut prng, NUM_ITEMS, MIN_VALUE, MAX_VALUE, MIN_WEIGHT, MAX_WEIGHT,
    );
    let allowed_weight = sum_weights(&mut items, true) / 2;

    // Display basic parameters.
    println!("*** Parameters ***");
    println!("# items:        {}", NUM_ITEMS);
    println!("Total value:    {}", sum_values(&mut items, true));
    println!("Total weight:   {}", sum_weights(&mut items, true));
    println!("Allowed weight: {}", allowed_weight);
    println!();

    // Exhaustive search
    if NUM_ITEMS > 23 {
        // Only run exhaustive search if num_items is small enough.
        println!("Too many items for exhaustive search\n");
    } else {
        println!("*** Exhaustive Search ***");
        run_algorithm(&exhaustive_search, &mut items, allowed_weight);
    }

    // Branch and Bound search
    if NUM_ITEMS > 40 {
        println!("Too many items for Branch and Bound search\n");
    } else {
        println!("*** Branch and Bound ***");
        run_algorithm(&branch_and_bound, &mut items, allowed_weight);
    }

    // Rod's technique
    if NUM_ITEMS > 60 {
        // Only run Rod's technique if num_items is small enough.
        println!("Too many items for Rod's technique\n");
    } else {
        println!("*** Rod's Technique ***");
        run_algorithm(&rods_technique, &mut items, allowed_weight);
    }

    // Rod's technique
    if NUM_ITEMS > 200 {
        // Only run Rod's technique if num_items is small enough.
        println!("Too many items for Rod's technique sorted\n");
    } else {
        println!("*** Rod's Technique Sorted ***");
        run_algorithm(&rods_technique_sorted, &mut items, allowed_weight);
    }

    // Dynamic programming
    println!("*** Dynamic programming ***");
    run_algorithm(&dynamic_programming, &mut items, allowed_weight);
}
