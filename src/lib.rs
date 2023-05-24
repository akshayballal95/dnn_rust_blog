use ndarray::prelude::*;
use num_integer::Roots;
use polars::prelude::*;
use rand::distributions::Uniform;
use rand::prelude::*;
use std::collections::HashMap;
use std::f32::consts::E;
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct LinearCache {
    pub a: Array2<f32>,
    pub w: Array2<f32>,
    pub b: Array2<f32>,
}

#[derive(Clone, Debug)]
pub struct ActivationCache {
    pub z: Array2<f32>,
}

#[derive(Clone, Debug)]
/// A struct that holds the parameters of a deep neural network.
pub struct DeepNeuralNetwork {
    pub layers: Vec<usize>,
    pub learning_rate: f32,
}

pub fn dataframe_from_csv(file_path: PathBuf) -> PolarsResult<(DataFrame, DataFrame)> {
    let data = CsvReader::from_path(file_path)?.has_header(true).finish()?;

    let training_dataset = data.drop("y")?;
    let training_labels = data.select(["y"])?;

    return Ok((training_dataset, training_labels));
}

pub fn array_from_dataframe(df: &DataFrame) -> Array2<f32> {
    df.to_ndarray::<Float32Type>().unwrap().reversed_axes()
}

trait Log {
    fn log(&self) -> Array2<f32>;
}

impl Log for Array2<f32> {
    fn log(&self) -> Array2<f32> {
        self.mapv(|x| x.log(std::f32::consts::E))
    }
}

pub fn sigmoid(z: &f32) -> f32 {
    1.0 / (1.0 + E.powf(-z))
}

pub fn sigmoid_activation(z: Array2<f32>) -> (Array2<f32>, ActivationCache) {
    (z.mapv(|x| sigmoid(&x)), ActivationCache { z })
}

pub fn relu_activation(z: Array2<f32>) -> (Array2<f32>, ActivationCache) {
    (z.mapv(|x| relu(&x)), ActivationCache { z })
}

pub fn relu(z: &f32) -> f32 {
    match *z > 0.0 {
        true => *z,
        false => 0.0,
    }
}

pub fn sigmoid_prime(z: &f32) -> f32 {
    sigmoid(z) * (1.0 - sigmoid(z))
}

pub fn relu_prime(z: &f32) -> f32 {
    match *z > 0.0 {
        true => 1.0,
        false => 0.0,
    }
}

pub fn sigmoid_backward(da: &Array2<f32>, activation_cache: ActivationCache) -> Array2<f32> {
    da * activation_cache.z.mapv(|x| sigmoid_prime(&x))
}

pub fn relu_backward(da: &Array2<f32>, activation_cache: ActivationCache) -> Array2<f32> {
    da * activation_cache.z.mapv(|x| relu_prime(&x))
}

pub fn linear_forward(
    a: &Array2<f32>,
    w: &Array2<f32>,
    b: &Array2<f32>,
) -> (Array2<f32>, LinearCache) {
    let z = w.dot(a) + b;

    let cache = LinearCache {
        a: a.clone(),
        w: w.clone(),
        b: b.clone(),
    };
    return (z, cache);
}

pub fn linear_forward_activation(
    a: &Array2<f32>,
    w: &Array2<f32>,
    b: &Array2<f32>,
    activation: &str,
) -> Result<(Array2<f32>, (LinearCache, ActivationCache)), String> {
    match activation {
        "sigmoid" => {
            let (z, linear_cache) = linear_forward(a, w, b);
            let (a_next, activation_cache) = sigmoid_activation(z);
            return Ok((a_next, (linear_cache, activation_cache)));
        }
        "relu" => {
            let (z, linear_cache) = linear_forward(a, w, b);
            let (a_next, activation_cache) = relu_activation(z);
            return Ok((a_next, (linear_cache, activation_cache)));
        }
        _ => return Err("wrong activation string".to_string()),
    }
}

pub fn linear_backward(
    dz: &Array2<f32>,
    linear_cache: LinearCache,
) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    let (a_prev, w, _b) = (linear_cache.a, linear_cache.w, linear_cache.b);
    let m = a_prev.shape()[1] as f32;
    let dw = (1.0 / m) * (dz.dot(&a_prev.reversed_axes()));
    let db_vec = ((1.0 / m) * dz.sum_axis(Axis(1))).to_vec();
    let db = Array2::from_shape_vec((db_vec.len(), 1), db_vec).unwrap();
    let da_prev = w.reversed_axes().dot(dz);

    (da_prev, dw, db)
}

pub fn linear_backward_activation(
    da: &Array2<f32>,
    cache: (LinearCache, ActivationCache),
    activation: &str,
) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    let (linear_cache, activation_cache) = cache;

    match activation {
        "sigmoid" => {
            let dz = sigmoid_backward(da, activation_cache);
            linear_backward(&dz, linear_cache)
        }
        "relu" => {
            let dz = relu_backward(da, activation_cache);
            linear_backward(&dz, linear_cache)
        }
        _ => panic!("wrong activation string"),
    }
}

impl DeepNeuralNetwork {
    /// Initializes the parameters of the neural network.
    ///
    /// ### Returns
    /// a Hashmap dictionary of randomly initialized weights and biases.
    pub fn initialize_parameters(&self) -> HashMap<String, Array2<f32>> {
        let between = Uniform::from(-1.0..1.0); // random number between -1 and 1
        let mut rng = rand::thread_rng(); // random number generator

        let number_of_layers = self.layers.len();

        let mut parameters: HashMap<String, Array2<f32>> = HashMap::new();

        // start the loop from the first hidden layer to the output layer.
        // We are not starting from 0 because the zeroth layer is the input layer.
        for l in 1..number_of_layers {
            let weight_array: Vec<f32> = (0..self.layers[l] * self.layers[l - 1])
                .map(|_| between.sample(&mut rng))
                .collect();

            let bias_array: Vec<f32> = (0..self.layers[l]).map(|_| 0.0).collect();

            let weight_matrix =
                Array2::from_shape_vec((self.layers[l], self.layers[l - 1]), weight_array).unwrap()
                    / (self.layers[l - 1]).sqrt() as f32;
            let bias_matrix = Array2::from_shape_vec((self.layers[l], 1), bias_array).unwrap();

            let weight_string = ["W", &l.to_string()].join("").to_string();
            let biases_string = ["b", &l.to_string()].join("").to_string();

            parameters.insert(weight_string, weight_matrix);
            parameters.insert(biases_string, bias_matrix);
        }
        parameters
    }

    pub fn forward(
        &self,
        x: &Array2<f32>,
        parameters: &HashMap<String, Array2<f32>>,
    ) -> (Array2<f32>, HashMap<String, (LinearCache, ActivationCache)>) {
        let number_of_layers = self.layers.len() - 1;

        let mut a = x.clone();
        let mut caches = HashMap::new();

        for l in 1..number_of_layers {
            let w_string = ["W", &l.to_string()].join("").to_string();
            let b_string = ["b", &l.to_string()].join("").to_string();

            let w = &parameters[&w_string];
            let b = &parameters[&b_string];

            let (a_temp, cache_temp) = linear_forward_activation(&a, w, b, "relu").unwrap();

            a = a_temp;

            caches.insert(l.to_string(), cache_temp);
        }

        // Compute activation of last layer with sigmoid
        let weight_string = ["W", &(number_of_layers).to_string()].join("").to_string();
        let bias_string = ["b", &(number_of_layers).to_string()].join("").to_string();

        let w = &parameters[&weight_string];
        let b = &parameters[&bias_string];

        let (al, cache) = linear_forward_activation(&a, w, b, "sigmoid").unwrap();
        caches.insert(number_of_layers.to_string(), cache);

        return (al, caches);
    }

    pub fn cost(&self, al: &Array2<f32>, y: &Array2<f32>) -> f32 {
        let m = y.shape()[1] as f32;
        let cost = -(1.0 / m)
            * (y.dot(&al.clone().reversed_axes().log())
                + (1.0 - y).dot(&(1.0 - al).reversed_axes().log()));

        return cost.sum();
    }

    pub fn backward(
        &self,
        al: &Array2<f32>,
        y: &Array2<f32>,
        caches: HashMap<String, (LinearCache, ActivationCache)>,
    ) -> HashMap<String, Array2<f32>> {
        let mut grads = HashMap::new();
        let num_of_layers = self.layers.len() - 1;

        let dal = -(y / al - (1.0 - y) / (1.0 - al));

        let current_cache = caches[&num_of_layers.to_string()].clone();
        let (mut da_prev, mut dw, mut db) =
            linear_backward_activation(&dal, current_cache, "sigmoid");

        let weight_string = ["dW", &num_of_layers.to_string()].join("").to_string();
        let bias_string = ["db", &num_of_layers.to_string()].join("").to_string();
        let activation_string = ["dA", &num_of_layers.to_string()].join("").to_string();

        grads.insert(weight_string, dw);
        grads.insert(bias_string, db);
        grads.insert(activation_string, da_prev.clone());

        for l in (1..num_of_layers).rev() {
            let current_cache = caches[&l.to_string()].clone();
            (da_prev, dw, db) = linear_backward_activation(&da_prev, current_cache, "relu");

            let weight_string = ["dW", &l.to_string()].join("").to_string();
            let bias_string = ["db", &l.to_string()].join("").to_string();
            let activation_string = ["dA", &l.to_string()].join("").to_string();

            grads.insert(weight_string, dw);
            grads.insert(bias_string, db);
            grads.insert(activation_string, da_prev.clone());
        }

        grads
    }
}
