use crate::error::errors::ShapeError;
use ndarray::Array2;

pub fn assert_same_shape_v2(a: &Array2<f64>, b: &Array2<f64>) -> Result<(), ShapeError> {
    if a.shape() != b.shape() {
        return Err(ShapeError {
            expected: a.shape().to_vec(),
            found: b.shape().to_vec(),
        });
    }

    Ok(())
}
