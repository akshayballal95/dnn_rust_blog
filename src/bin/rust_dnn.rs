use dnn_rust_blog::*;
use std::env;
fn main(){
    env::set_var("RUST_BACKTRACE", "1");
    let neural_network_layers:Vec<usize> = vec![12288,3, 5, 10, 1];
    let learning_rate = 0.001;

    let (training_set, training_labels) = dataframe_from_csv("datasets/training_set.csv".into()).unwrap();

    let training_set_array = array_from_dataframe(&training_set);
    let training_labels_array = array_from_dataframe(&training_labels);


    let model = DeepNeuralNetwork{
        layers : neural_network_layers,
        learning_rate
    };

    let parameters = model.initialize_parameters();
    let (al, caches) = model.forward(&training_set_array, &parameters);
    let backward = model.backward(&al, &training_labels_array, caches);
    

}
