use dnn_rust_blog::*;
fn main(){
    let neural_network_layers:Vec<usize> = vec![12288,3, 5, 10, 1];
    let learning_rate = 0.001;

    let (training_set, training_labels) = dataframe_from_csv("datasets/training_set.csv".into()).unwrap();

    let training_set_array = array_from_dataframe(&training_set);
    let training_labels_array = array_from_dataframe(&training_labels).reversed_axes();


    let model = DeepNeuralNetwork{
        layers : neural_network_layers,
        learning_rate
    };

    model.initialize_parameters();
    

}
