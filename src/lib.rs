mod error;
mod layers;
mod loss;
mod nn;
mod operations;
mod regression;
mod utils;

#[cfg(test)]
mod tests {
    use crate::operations::bias_add::BiasAdd;
    use crate::operations::operation::Operation;
    use crate::operations::sigmoid::Sigmoid;
    use crate::operations::weight_multiply::WeightMultiply;
    use ndarray::Array2;

    #[test]
    fn operation_test() {
        let weights = Array2::from_elem((3, 2), 0.5);
        let biases = Array2::from_elem((1, 2), 0.1);
        let weight_multiply_op = WeightMultiply::new(weights);
        let bias_add_op = BiasAdd::new(biases);
        let sigmoid_op = Sigmoid::new();

        let mut operations: Vec<Box<dyn Operation>> = Vec::new();

        operations.push(Box::new(weight_multiply_op));
        operations.push(Box::new(bias_add_op));
        operations.push(Box::new(sigmoid_op));

        // Now you can iterate over operations and call methods defined by the NeuralOperation trait.
        let input_data = Array2::from_elem((1, 3), 1.0);
        let mut current_input = input_data;

        for op in operations.iter_mut() {
            match op.forward(current_input.clone()) {
                Ok(output) => current_input = output,
                Err(e) => println!("Error during forward pass: {}", e),
            }
        }

        // Similarly, you can perform a backward pass with a gradient.
        let gradient = Array2::from_elem((1, 2), 1.0);
        let mut current_gradient = gradient;

        for op in operations.iter_mut().rev() {
            match op.backward(current_gradient.clone()) {
                Ok(output_grad) => current_gradient = output_grad,
                Err(e) => println!("Error during backward pass: {}", e),
            }
        }
    }
}
