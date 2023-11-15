use crate::layers::layer::Layer;
use crate::loss::loss::Loss;
use ndarray::Array2;
use std::cell::RefCell;
use std::rc::Rc;

pub struct NeuralNetwork {
    pub layers: Vec<Rc<RefCell<dyn Layer>>>,
    pub loss: Rc<RefCell<dyn Loss>>,
    pub seed: Option<f64>,
}

impl NeuralNetwork {
    pub fn new(
        layers: Vec<Rc<RefCell<dyn Layer>>>,
        loss: Rc<RefCell<dyn Loss>>,
        seed: Option<f64>,
    ) -> Self {
        Self { layers, loss, seed }
    }

    pub fn forward(&mut self, x_batch: Array2<f64>) -> Array2<f64> {
        // Passes data forward through a series of layers.
        let mut x_out = x_batch;
        for layer in &mut self.layers {
            x_out = layer.borrow_mut().forward(x_out);
        }
        x_out
    }

    pub fn backward(&mut self, loss_grad: Array2<f64>) {
        // Passes data backward through a series of layers.
        let mut grad = loss_grad;
        for layer in self.layers.iter_mut().rev() {
            grad = layer.borrow_mut().backward(grad);
        }
    }

    pub fn train_batch(&mut self, x_batch: Array2<f64>, y_batch: Array2<f64>) -> f64 {
        // Passes data forward through the layers.
        let predictions = self.forward(x_batch);
        // Computes the loss.
        let loss = self.loss.borrow_mut().forward(&predictions, &y_batch);
        // Computes the gradient of the loss.
        let loss_grad = self.loss.borrow_mut().backward();
        // Passes the gradient of the loss backward through the layers.
        self.backward(loss_grad);
        loss
    }

    pub fn params(&mut self) -> Vec<Array2<f64>> {
        // Collects the parameters from the layers.
        let mut params = Vec::new();
        for layer in &mut self.layers {
            params.append(&mut layer.borrow_mut().params());
        }
        params
    }

    pub fn params_grad(&mut self) -> Vec<Array2<f64>> {
        // Collects the gradients from the layers.
        let mut param_grads = Vec::new();
        for layer in &mut self.layers {
            param_grads.append(&mut layer.borrow_mut().params_grad());
        }
        param_grads
    }
}

#[cfg(test)]
mod tests {
    use crate::dense_layer;
    use crate::layers::dense::Dense;
    use crate::loss::mse::MeanSquaredError;
    use crate::nn::neural_network::NeuralNetwork;
    use crate::operations::sigmoid::Sigmoid;
    use ndarray::array;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn test_linear_regression() {
        let layer = Rc::new(RefCell::new(Dense::new(2, Box::new(Sigmoid::new()))));
        let layer = dense_layer!(2, Sigmoid);
        let loss = Rc::new(RefCell::new(MeanSquaredError::new()));
        let mut nn = NeuralNetwork::new(vec![layer], loss, None);

        let x_batch = array![[1.0, 2.0]];
        let y_batch = array![[0.5, 1.5]];

        let loss = nn.train_batch(x_batch, y_batch);
        println!("Loss: {}", loss);
    }

    #[test]
    fn test_nn() {
        let layer1 = Rc::new(RefCell::new(Dense::new(12, Box::new(Sigmoid::new()))));
        let layer2 = Rc::new(RefCell::new(Dense::new(2, Box::new(Sigmoid::new()))));
        let layer1 = dense_layer!(12, Sigmoid);
        let layer2 = dense_layer!(2, Sigmoid);

        let loss = Rc::new(RefCell::new(MeanSquaredError::new()));
        let mut nn = NeuralNetwork::new(vec![layer1, layer2], loss, None);

        let x_batch = array![[1.0, 2.0]];
        let y_batch = array![[0.5, 1.5]];

        let loss = nn.train_batch(x_batch, y_batch);
        println!("Loss: {}", loss);
    }
}
