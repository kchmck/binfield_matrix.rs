//! Vector-matrix multiplication for [GF(2)](http://mathworld.wolfram.com/FiniteField.html)
//! binary field error correction codes.
//!
//! These routines calculate the multiplication **vM**<sup>T</sup> = **Mv**<sup>T</sup> of
//! a 1×M binary vector **v** with an N×M binary matrix **M**, using GF(2) addition and
//! multiplication. The input vector, output vector, and matrix columns are represented as
//! binary words, so the maximum vector size is determined by the maximum machine word
//! size.
//!
//! ## Example
//!
//! The following examples mutiply the codeword `1010` by the matrix
//!
//! ```text
//! 1 1 1 1
//! 0 0 1 0
//! 1 0 0 0
//! 0 1 0 1
//! 0 0 1 0
//! 1 0 1 0
//! ```
//!
//! The first example computes only the parity bits, and the second example computes the
//! "systematic" codeword, which appends the parity bits to the original codeword.
//!
//! ```rust
//! use binfield_matrix::{matrix_mul, matrix_mul_systematic};
//!
//! assert_eq!(matrix_mul::<u32, u32>(0b1010, &[
//!     0b1111,
//!     0b0010,
//!     0b1000,
//!     0b0101,
//!     0b0010,
//!     0b1010,
//! ]), 0b011010);
//!
//! assert_eq!(matrix_mul_systematic::<u32, u32>(0b1010, &[
//!     0b1111,
//!     0b0010,
//!     0b1000,
//!     0b0101,
//!     0b0010,
//!     0b1010,
//! ]), 0b1010011010);
//! ```

extern crate num_traits;

use num_traits::PrimInt;

/// Compute **vM**<sup>T</sup>, where **v** is the given word and **M** is the given
/// matrix.
pub fn matrix_mul<I, O>(word: I, mat: &[I]) -> O where
    I: PrimInt,
    O: PrimInt + From<u8>,
{
    accum_rows(word, O::zero(), mat)
}

/// Compute [ **v** | **vM**<sup>T</sup> ], where **v** is the given word and **M** is the
/// given matrix.
pub fn matrix_mul_systematic<I, O>(word: I, mat: &[I]) -> O where
    I: PrimInt + Into<O>,
    O: PrimInt + From<u8>,
{
    accum_rows(word, word.into(), mat)
}

/// Starting with the given initial accumulator, compute the GF(2) dot product of the
/// given word with each row in the given matrix, shifting each resulting bit into the LSB
/// of the accumulator.
///
/// This effectively computes the vector-matrix multiplication.
fn accum_rows<I, O>(word: I, init: O, mat: &[I]) -> O where
    I: PrimInt,
    O: PrimInt + From<u8>,
{
    mat.iter().fold(init, |accum, &row| {
        let bit = ((word & row).count_ones() & 1) as u8;
        accum << 1 | bit.into()
    })
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_mul() {
        let w: u16 = matrix_mul(0b0110011, &[
            0b1010101,
            0b0110011,
            0b0001111,
        ]);
        assert_eq!(w, 0b000);

        let w: u16 = matrix_mul(0b0110111, &[
            0b1010101,
            0b0110011,
            0b0001111,
        ]);
        assert_eq!(w, 0b101);
    }

    #[test]
    fn test_mul_sys() {
        let w: u16 = matrix_mul_systematic(0u16, &[
            0b11111110000,
            0b11110001110,
            0b11001101101,
            0b10101011011,
        ]);
        assert_eq!(w, 0);

        let w: u16 = matrix_mul_systematic(0b11111111111u16, &[
            0b11111110000,
            0b11110001110,
            0b11001101101,
            0b10101011011,
        ]);
        assert_eq!(w, 0b11111111111_1111);

        let w: u16 = matrix_mul_systematic(0b11111111101u16, &[
            0b11111110000,
            0b11110001110,
            0b11001101101,
            0b10101011011,
        ]);
        assert_eq!(w, 0b11111111101_1010);
    }
}
