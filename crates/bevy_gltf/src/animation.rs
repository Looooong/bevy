
use bevy_asset::{Handle};
use bevy_ecs::prelude::ReflectComponent;
use bevy_math::{Quat, Vec3};
use gltf::animation::{Interpolation, util::{MorphTargetWeights, ReadOutputs, Rotations}};

use super::{Gltf, GltfNode};
use bevy_reflect::{Reflect, TypeUuid};

/// A component for transform nodes loaded by the Gltf loader that indicates this entity is the target of at least one animation in the Gltf asset.
///
/// One or more animations may target the same node at different property paths. Each unique animation/channel pair  that targets this entity receives an entry in the `animations` and `channel_indices` vectors. For example, the animation index at animations\[`1`\] targets a property of this entity through the channel index identified by the value at channel_indices\[`1`\].
#[derive(Debug, Default, Clone, Reflect)]
#[reflect(Component)]
pub struct GltfAnimTargetInfo {
    pub gltf: Handle<Gltf>,
    pub animation_indices: Vec<usize>,
    pub channel_indices: Vec<usize>,
}

/// Contains a collection of animation channels, each of which targets a single glTF-animatable property (position, rotation, scale, morph target weight) and sampling data.
#[derive(Debug, Clone, TypeUuid)]
#[uuid = "cc71ba69-fc20-4665-b399-27da45618653"]
pub struct GltfAnimation {
    /// Animations target node properties through channels, each of which contains a target node, property, keyframe times, and per-keyframe values.
    pub channels: Vec<GltfAnimChannel>,
    /// The index of this animation in the Gltf.
    pub index: usize,
    /// The name of this animation. Animations are not guaranteed to have names.
    pub name: Option<String>,
    pub(crate) start_time: f32,
    pub(crate) end_time: f32,
}
impl GltfAnimation {
    /// The time in seconds of the earliest keyframe among all channels in this animation.
    pub fn start_time(&self) -> f32 { self.start_time }
    /// The time in seconds of the latest keyframe among all channels in this animation.
    pub fn end_time(&self) -> f32 { self.end_time }
    /// The duration in seconds between the earliest and latest keyframes among all channels in this animation.
    pub fn duration(&self) -> f32 { self.end_time - self.start_time }
}

/// Targets a single glTF-animatable property of a glTF node (position, rotation, scale, or morph target weight) and sampling data for converting animation time in seconds to the animated property value.
#[derive(Debug, Clone)]
pub struct GltfAnimChannel {
    pub target: GltfAnimTarget,
    pub sampler: GltfAnimSampler,
    pub index: usize,
    pub(crate) start_time: f32,
    pub(crate) end_time: f32, 
}
impl GltfAnimChannel {
    /// The time in seconds of first keyframe in this channel.
    pub fn start_time(&self) -> f32 { self.start_time }
    /// The time in seconds of last keyframe in this channel.
    pub fn end_time(&self) -> f32 { self.end_time }
    /// The duration in seconds between the first and last keyframes in this channel.
    pub fn duration(&self) -> f32 { self.end_time - self.start_time }
}

/// Contains a handle to the target GltfNode for animation and the animation path for the node (translation, rotation, scale, or morph target weight).
#[derive(Debug, Clone)]
pub struct GltfAnimTarget {
    pub node: Handle<GltfNode>,
    pub path: GltfAnimTargetProperty
}

/// As gltf_json::Property. Specifies a property to animate. Valid target properties are position, rotation, scale, or morph target weights.
#[derive(Debug, Clone)]
pub enum GltfAnimTargetProperty {
    /// XYZ translation vector.
    Position,

    /// XYZW rotation quaternion.
    Rotation,

    /// XYZ scale vector.
    Scale,

    /// Weights of morph targets.
    MorphTargetWeights
}

impl From<gltf::animation::Property> for GltfAnimTargetProperty {
    fn from(property: gltf::animation::Property) -> Self {
        match property {
            gltf::animation::Property::Translation => GltfAnimTargetProperty::Position,
            gltf::animation::Property::Rotation => GltfAnimTargetProperty::Rotation,
            gltf::animation::Property::Scale => GltfAnimTargetProperty::Scale,
            gltf::animation::Property::MorphTargetWeights => GltfAnimTargetProperty::MorphTargetWeights,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GltfAnimSampler {
    pub input: GltfAnimKeyframeTimes,
    pub interpolation: GltfAnimInterpolation,
    pub output: GltfAnimOutputValues
}

#[derive(Debug, Clone)]
pub struct GltfAnimKeyframeTimes(pub Vec<f32>);

#[derive(Debug, Clone)]
pub enum GltfAnimInterpolation {
  Linear,
  Step,
  CubicSpline
}

impl From<gltf::animation::Interpolation> for GltfAnimInterpolation {
    fn from(interp: gltf::animation::Interpolation) -> Self {
        match interp {
            Interpolation::Linear => GltfAnimInterpolation::Linear,
            Interpolation::Step => GltfAnimInterpolation::Step,
            Interpolation::CubicSpline => GltfAnimInterpolation::CubicSpline,
        }
    }
}

#[derive(Debug, Clone)]
pub enum GltfAnimOutputValues {
  Translations(Vec<Vec3>),
  Rotations(Vec<Quat>),
  Scales(Vec<Vec3>),
  MorphTargetWeights(Vec<f32>)
}

impl GltfAnimOutputValues {
    pub fn len(&self) -> usize {
        match self {
            GltfAnimOutputValues::Translations(vec) => vec.len(),
            GltfAnimOutputValues::Rotations(vec) => vec.len(),
            GltfAnimOutputValues::Scales(vec) => vec.len(),
            GltfAnimOutputValues::MorphTargetWeights(vec) => vec.len(),
        }
    }
}

impl From<ReadOutputs<'_>> for GltfAnimOutputValues {
    fn from(outputs: ReadOutputs<'_>) -> Self {
        match outputs {
            ReadOutputs::Translations(translations) => GltfAnimOutputValues::Translations(
                translations.map(|xyz| xyz.into()).collect()
            ),
            ReadOutputs::Rotations(rotations) => GltfAnimOutputValues::Rotations(
                match rotations {
                    // glTF reference for converting non-float encoded types in quaternions to float and back:
                    // https://github.com/KhronosGroup/glTF/blob/master/specification/2.0/README.md#animations
                    //
                    // The encoding is always XYZW.
                    Rotations::I8(i8x4s) => i8x4s
                        .map(|i8x4| i8x4.map(|c| gltf_i8_to_f32(c)))
                        .map(|xyzw| Quat::from_xyzw(xyzw[0], xyzw[1], xyzw[2], xyzw[3]))
                        .collect(),
                    Rotations::U8(u8x4s) => u8x4s
                        .map(|u8x4| u8x4.map(|c| gltf_u8_to_f32(c)))
                        .map(|xyzw| Quat::from_xyzw(xyzw[0], xyzw[1], xyzw[2], xyzw[3]))
                        .collect(),
                    Rotations::I16(i16x4s) => i16x4s
                        .map(|i16x4| i16x4.map(|c| gltf_i16_to_f32(c)))
                        .map(|xyzw| Quat::from_xyzw(xyzw[0], xyzw[1], xyzw[2], xyzw[3]))
                        .collect(),
                    Rotations::U16(u16x4s) => u16x4s
                        .map(|u16x4| u16x4.map(|c| gltf_u16_to_f32(c)))
                        .map(|xyzw| Quat::from_xyzw(xyzw[0], xyzw[1], xyzw[2], xyzw[3]))
                        .collect(),
                    Rotations::F32(f32x4s) => f32x4s
                        .map(|xyzw| Quat::from_xyzw(xyzw[0], xyzw[1], xyzw[2], xyzw[3]))
                        .collect(),
                }
            ),
            ReadOutputs::Scales(scales) => GltfAnimOutputValues::Scales(
                scales.map(|xyz| xyz.into()).collect()
            ),
            ReadOutputs::MorphTargetWeights(weights) => GltfAnimOutputValues::MorphTargetWeights(
                match weights {
                    MorphTargetWeights::I8(i8s) => i8s
                        .map(|i8| gltf_i8_to_f32(i8))
                        .collect(),
                    MorphTargetWeights::U8(u8s) => u8s
                        .map(|u8| gltf_u8_to_f32(u8))
                        .collect(),
                    MorphTargetWeights::I16(i16s) => i16s
                        .map(|i16| gltf_i16_to_f32(i16))
                        .collect(),
                    MorphTargetWeights::U16(u16s) => u16s
                        .map(|u16| gltf_u16_to_f32(u16))
                        .collect(),
                    MorphTargetWeights::F32(f32s) => f32s
                        .collect(),
                }
            ),
        }
    }
}

/// glTF reference for converting non-float encoded types in quaternions to float and back:
// https://github.com/KhronosGroup/glTF/blob/master/specification/2.0/README.md#animations

/// Converts from i8 to f32 according to the glTF specification for animation data values.
pub fn gltf_i8_to_f32(i: i8) -> f32 { (i as f32 / i8::MAX as f32).max(-1.0) }

/// Converts from u8 to f32 according to the glTF specification for animation data values.
pub fn gltf_u8_to_f32(u: u8) -> f32 { u as f32 / u8::MAX as f32 }

/// Converts from i16 to f32 according to the glTF specification for animation data values.
pub fn gltf_i16_to_f32(i: i16) -> f32 { (i as f32 / i16::MAX as f32).max(-1.0) }

/// Converts from u16 to f32 according to the glTF specification for animation data values.
pub fn gltf_u16_to_f32(u: u16) -> f32 { u as f32 / u16::MAX as f32 }
