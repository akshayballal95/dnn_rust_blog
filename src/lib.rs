use ndarray::prelude::*;
use polars::prelude::*;
use rand::distributions::Uniform;
use rand::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct LinearCache {
    pub a: Array2<f32>,
    pub w: Array2<f32>,
    pub b: Array2<f32>,
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

pub fn linear_forward(a: &Array2<f32>, w: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {

    let z = w.dot(a) + b;

    let cache = LinearCache {
        a: a.clone(),
        w: w.clone(),
        b: b.clone(),
    };
    return z

}

pub fn linear_forward_activation(z:&Array2<f32>, activation:&str) -> Array2<f32> {
    match activation {
        "sigmoid" => {
            let a = sigmoid(z);
            return a
            },
        "relu" => {
            let a = relu(z);
            return a;
        }
        _ => panic!("Unsupported activation function"),

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
            let weight_array: Vec<f32> = (0..self.layers[l]*self.layers[l-1])
                .map(|_| between.sample(&mut rng))
                .collect();

            let bias_array: Vec<f32> = (0..self.layers[l]).map(|_| 0.0).collect();


            let weight_matrix =
                Array::from_shape_vec((self.layers[l], self.layers[l - 1]), weight_array).unwrap();
            let bias_matrix = Array::from_shape_vec((self.layers[l], 1), bias_array).unwrap();

            let weight_string = ["W", &l.to_string()].join("").to_string();
            let biases_string = ["b", &l.to_string()].join("").to_string();

            parameters.insert(weight_string, weight_matrix);
            parameters.insert(biases_string, bias_matrix);
        }
        parameters
    }

    pub fn forward(&self, x: &Array2<f32>, parameters: &HashMap<String, Array2<f32>>) -> Array2<f32> {
        let number_of_layers = parameters.len()-1;

        let a = x; 
        let mut caches = HashMap::new();

        for l in 1..number_of_layers {
            let w_string = ["W", &l.to_string()].join("").to_string();
            let b_string = ["b", &l.to_string()].join("").to_string();

            let w = parameters[&w_string];
            let b = parameters[&b_string];

            
        }
        return x*
    }
}
