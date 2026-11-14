// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Binding of the tensors passed to `decode` to the decoder's output roles.
//!
//! The decode functions find each role's tensor by shape, taking the first
//! that fits. That is exact while every configured output has its own shape
//! and wrong when two share one: a split detector with four classes has
//! scores shaped like its boxes, a segmentation model with as many classes
//! as protos has scores shaped like its mask coefficients.
//!
//! [`OutputBinding`] exists only for such a model. It picks each role's
//! tensor by the rule the semantic decoder uses
//! ([`select_output`](crate::semantic::select_output)): the tensor named
//! like the output, else the only one with its shape, else the one at the
//! output's configured position when every model output was passed. The
//! tensors are then handed on in the order the decode functions look roles
//! up, so each first-fit lookup lands on the tensor bound here.

use super::config::ConfigOutput;
use crate::DecoderError;
use edgefirst_tensor::TensorDyn;

/// A decoded output's role, declared in the order the decode functions look
/// roles up. Every function that reads two roles reads them in this order,
/// which is what lets a reordered tensor list stand in for a binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Role {
    Detection,
    Boxes,
    Scores,
    Classes,
    MaskCoefficients,
    Protos,
    Segmentation,
}

impl Role {
    /// The role `output` is decoded as, or `None` for a `masks` output,
    /// which no decode function reads.
    pub(crate) fn of(output: &ConfigOutput) -> Option<Self> {
        match output {
            ConfigOutput::Detection(_) => Some(Role::Detection),
            ConfigOutput::Boxes(_) => Some(Role::Boxes),
            ConfigOutput::Scores(_) => Some(Role::Scores),
            ConfigOutput::Classes(_) => Some(Role::Classes),
            ConfigOutput::MaskCoefficients(_) => Some(Role::MaskCoefficients),
            ConfigOutput::Protos(_) => Some(Role::Protos),
            ConfigOutput::Segmentation(_) => Some(Role::Segmentation),
            ConfigOutput::Mask(_) => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Role::Detection => "detection",
            Role::Boxes => "boxes",
            Role::Scores => "scores",
            Role::Classes => "classes",
            Role::MaskCoefficients => "mask_coefs",
            Role::Protos => "protos",
            Role::Segmentation => "segmentation",
        }
    }
}

/// One output of the model's configuration, in configured order.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ConfiguredOutput {
    /// The output's name in the model, where the configuration gives one.
    pub(crate) name: Option<String>,
    pub(crate) shape: Vec<usize>,
    /// `None` for an output the decoder does not read.
    pub(crate) role: Option<Role>,
}

impl ConfiguredOutput {
    /// An output of a configuration that names none.
    pub(crate) fn unnamed(output: &ConfigOutput) -> Self {
        Self {
            name: None,
            shape: output.shape().to_vec(),
            role: Role::of(output),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Slot {
    role: Role,
    name: Option<String>,
    shape: Vec<usize>,
    /// Index among the configured outputs.
    position: usize,
    /// The other configured outputs of this shape, for the error message.
    shared_with: String,
}

/// How the roles of a model whose outputs share shapes find their tensors.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct OutputBinding {
    /// The decoded outputs, in lookup order.
    slots: Vec<Slot>,
    /// Number of configured outputs, decoded or not.
    count: usize,
}

impl OutputBinding {
    /// The binding for `outputs`, or `None` when every decoded output has a
    /// shape of its own and lookup by shape is exact.
    pub(crate) fn new(outputs: &[ConfiguredOutput]) -> Option<Self> {
        let sharing = |index: usize| {
            outputs
                .iter()
                .enumerate()
                .filter(move |(other, o)| *other != index && o.shape == outputs[index].shape)
                .map(|(_, o)| o)
        };
        let ambiguous = outputs
            .iter()
            .enumerate()
            .any(|(index, o)| o.role.is_some() && sharing(index).next().is_some());
        if !ambiguous {
            return None;
        }
        let mut slots: Vec<Slot> = outputs
            .iter()
            .enumerate()
            .filter_map(|(position, o)| {
                let shared_with = sharing(position)
                    .map(|other| match (other.role, &other.name) {
                        (Some(role), _) => format!("`{}`", role.name()),
                        (None, Some(name)) => format!("`{name}`"),
                        (None, None) => "an output that is not decoded".to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join(" and ");
                Some(Slot {
                    role: o.role?,
                    name: o.name.clone(),
                    shape: o.shape.clone(),
                    position,
                    shared_with,
                })
            })
            .collect();
        // Stable: outputs of one role (ModelPack's per-scale detections)
        // keep their configured order.
        slots.sort_by_key(|slot| slot.role);
        Some(Self {
            slots,
            count: outputs.len(),
        })
    }

    /// The tensor of each decoded output, in lookup order.
    ///
    /// # Errors
    ///
    /// [`DecoderError::InvalidShape`] when no tensor has an output's shape,
    /// [`DecoderError::InvalidConfig`] when several do and neither a name
    /// nor a full set of outputs in configured order says which.
    pub(crate) fn bind<'a>(
        &self,
        outputs: &[&'a TensorDyn],
    ) -> Result<Vec<&'a TensorDyn>, DecoderError> {
        let mut taken = vec![false; outputs.len()];
        // Names are settled for every output before any falls back to shape
        // or position, so a fallback never takes a tensor another output
        // names.
        let mut bound: Vec<Option<usize>> = Vec::with_capacity(self.slots.len());
        for slot in &self.slots {
            let index = slot.named(outputs, &taken);
            if let Some(index) = index {
                taken[index] = true;
            }
            bound.push(index);
        }
        for (slot, index) in self.slots.iter().zip(&mut bound) {
            if index.is_none() {
                let found = self.unnamed(slot, outputs, &taken)?;
                taken[found] = true;
                *index = Some(found);
            }
        }
        Ok(bound.into_iter().flatten().map(|i| outputs[i]).collect())
    }

    /// The tensor for `slot` among those no output names: the only one of
    /// its shape, else the one at its configured position.
    fn unnamed(
        &self,
        slot: &Slot,
        outputs: &[&TensorDyn],
        taken: &[bool],
    ) -> Result<usize, DecoderError> {
        let fits = |index: usize| slot.fits(outputs, taken, index);
        let mut candidates = (0..outputs.len()).filter(|&i| fits(i));
        match (candidates.next(), candidates.next()) {
            (Some(index), None) => Ok(index),
            (Some(_), Some(_)) if outputs.len() == self.count && fits(slot.position) => {
                Ok(slot.position)
            }
            (Some(_), Some(_)) => Err(DecoderError::InvalidConfig(format!(
                "several outputs have the shape {:?} of the `{}` output, which {} shares; \
                 pass every model output in the configured order, or name the tensors",
                slot.shape,
                slot.role.name(),
                slot.shared_with,
            ))),
            (None, _) => Err(DecoderError::InvalidShape(format!(
                "no output has the shape {:?} of the `{}` output; got {:?}",
                slot.shape,
                slot.role.name(),
                outputs
                    .iter()
                    .map(|t| t.shape().to_vec())
                    .collect::<Vec<_>>(),
            ))),
        }
    }
}

impl Slot {
    fn fits(&self, outputs: &[&TensorDyn], taken: &[bool], index: usize) -> bool {
        !taken[index] && outputs[index].shape() == self.shape
    }

    /// The free tensor of this output's shape that carries its name.
    fn named(&self, outputs: &[&TensorDyn], taken: &[bool]) -> Option<usize> {
        let name = self.name.as_ref()?;
        (0..outputs.len()).find(|&i| self.fits(outputs, taken, i) && outputs[i].name() == *name)
    }
}

/// The order in which merged logical outputs, produced in schema order, are
/// handed to the decode functions: by role, so that two of one shape are
/// looked up as their roles and not as the schema happens to list them. An
/// output no function reads goes last, behind any decoded output of its
/// shape. `None` when the schema already lists them that way.
pub(crate) fn lookup_order(outputs: &[ConfigOutput]) -> Option<Vec<usize>> {
    let mut order: Vec<usize> = (0..outputs.len()).collect();
    order.sort_by_key(|&index| {
        let role = Role::of(&outputs[index]);
        (role.is_none(), role)
    });
    let reordered = order.iter().enumerate().any(|(at, &index)| at != index);
    reordered.then_some(order)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn output(name: Option<&str>, shape: &[usize], role: Option<Role>) -> ConfiguredOutput {
        ConfiguredOutput {
            name: name.map(str::to_string),
            shape: shape.to_vec(),
            role,
        }
    }

    #[test]
    fn outputs_with_their_own_shapes_need_no_binding() {
        let outputs = [
            output(None, &[1, 4, 8400], Some(Role::Boxes)),
            output(None, &[1, 80, 8400], Some(Role::Scores)),
        ];
        assert_eq!(OutputBinding::new(&outputs), None);
    }

    /// An output the decoder does not read still makes a decoded output of
    /// its shape ambiguous.
    #[test]
    fn an_undecoded_output_of_the_same_shape_needs_a_binding() {
        let outputs = [
            output(None, &[1, 4, 8400], Some(Role::Boxes)),
            output(Some("aux"), &[1, 4, 8400], None),
        ];
        let binding = OutputBinding::new(&outputs).expect("shapes collide");
        assert_eq!(binding.slots.len(), 1);
        assert_eq!(binding.count, 2);
        assert_eq!(binding.slots[0].shared_with, "`aux`");
    }

    /// A `masks` output is read by no decode function, so it has no slot and
    /// its tensor is never asked for.
    #[test]
    fn a_masks_output_has_no_slot() {
        let shape = vec![1, 8, 8, 4];
        let outputs = [
            ConfiguredOutput::unnamed(&ConfigOutput::Segmentation(crate::configs::Segmentation {
                shape: shape.clone(),
                ..Default::default()
            })),
            ConfiguredOutput::unnamed(&ConfigOutput::Mask(crate::configs::Mask {
                shape,
                ..Default::default()
            })),
        ];
        let binding = OutputBinding::new(&outputs).expect("shapes collide");
        assert_eq!(binding.slots.len(), 1);
        assert_eq!(binding.slots[0].role, Role::Segmentation);
        assert_eq!(binding.count, 2);
    }

    /// Merged outputs are looked up first-fit, so an unread `masks` output
    /// goes behind the segmentation whose shape it may share.
    #[test]
    fn an_unread_merged_output_is_ordered_last() {
        let outputs = [
            ConfigOutput::Mask(Default::default()),
            ConfigOutput::Segmentation(Default::default()),
        ];
        assert_eq!(lookup_order(&outputs), Some(vec![1, 0]));
    }

    /// Two undecoded outputs of one shape concern no role.
    #[test]
    fn undecoded_outputs_sharing_a_shape_need_no_binding() {
        let outputs = [
            output(None, &[1, 4, 8400], Some(Role::Boxes)),
            output(Some("a"), &[1, 7], None),
            output(Some("b"), &[1, 7], None),
        ];
        assert_eq!(OutputBinding::new(&outputs), None);
    }
}
