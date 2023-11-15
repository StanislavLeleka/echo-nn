use crate::nn::neural_network::NeuralNetwork;
use crate::optimizer::optimizer::Optimizer;
use ndarray::Array2;
use std::cell::RefCell;
use std::rc::Rc;

pub struct SGD {
    pub lr: f64,
    pub net: Rc<RefCell<NeuralNetwork>>,
}

impl SGD {
    pub fn new(lr: f64, net: Rc<RefCell<NeuralNetwork>>) -> SGD {
        SGD { lr, net }
    }
}

impl Optimizer for SGD {
    fn step(&self) {
        // Updates the parameters of the network.
        let mut net_borrowed = self.net.borrow_mut();
        let mut params = net_borrowed.params();
        let params_grad = net_borrowed.params_grad();

        // Iterate over each pair of parameter array and gradient array
        for (param, grad) in params.iter_mut().zip(params_grad.iter()) {
            // Perform the update element-wise
            param.zip_mut_with(grad, |p, &g| {
                *p -= g * self.lr;
            });
        }
    }
}
