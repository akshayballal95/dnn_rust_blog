use dnn_rust_blog::*;
use std::env;
fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    let neural_network_layers: Vec<usize> = vec![12288, 20, 7, 5, 1];
    let learning_rate = 0.0075;
    let iterations = 1000;

    let (training_data, training_labels) =
        dataframe_from_csv("datasets/training_set.csv".into()).unwrap();
    let (test_data, test_labels) = dataframe_from_csv("datasets/test_set.csv".into()).unwrap();

    let training_data_array = array_from_dataframe(&training_data)/255.0;
    let training_labels_array = array_from_dataframe(&training_labels);
    let test_data_array = array_from_dataframe(&test_data)/255.0;
    let test_labels_array = array_from_dataframe(&test_labels);

    let model = DeepNeuralNetwork {
        layers: neural_network_layers,
        learning_rate,
    };

    let parameters = model.initialize_parameters();

    let parameters = model.train_model(
        &training_data_array,
        &training_labels_array,
        parameters,
        iterations,
        model.learning_rate,
    );
    write_parameters_to_json_file(&parameters, "model.json".into());

    let training_predictions = model.predict(&training_data_array, &parameters);
    println!(
        "Training Set Accuracy: {}%",
        model.score(&training_predictions, &training_labels_array)
    );

    let test_predictions = model.predict(&test_data_array, &parameters);
    println!(
        "Test Set Accuracy: {}%",
        model.score(&test_predictions, &test_labels_array)
    );
}
