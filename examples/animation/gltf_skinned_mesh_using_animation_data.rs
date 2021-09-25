use bevy::{
    gltf::{
        Gltf, GltfAnimInterpolation, GltfAnimOutputValues, GltfAnimSampler, GltfAnimTargetInfo,
        GltfAnimTargetProperty, GltfAnimation,
    },
    pbr::AmbientLight,
    prelude::*,
    utils::HashMap,
};

/// Skinned mesh example with mesh, joints, and animation data loaded from a glTF file.
/// Example taken from https://github.com/KhronosGroup/glTF-Tutorials/blob/master/gltfTutorial/gltfTutorial_019_SimpleSkin.md
fn main() {
    App::build()
        .add_plugins(DefaultPlugins)
        .insert_resource(AmbientLight {
            brightness: 1.0,
            ..Default::default()
        })
        .add_startup_system(setup.system())
        // This general-purpose system adds AnimationControllers to entities that request them for a given Gltf.
        .add_system(activate_animations.system())
        // This example system updates the GltfAnimationController to set animation times and weights. The GltfAnimationController can be used with any Gltf animation data to drive and blend multiple animations.
        .add_system(update_animation_controllers.system())
        // This general-purpose system takes the times and weights specified by GltfAnimationControllers and updates entities that automatically receive GltfAnimTargetInfo from the Gltf animation loader when they are spawned.
        .add_system(update_gltf_animations.system())
        .run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    // Create a camera
    let mut camera = OrthographicCameraBundle::new_2d();
    camera.orthographic_projection.near = -1.0;
    camera.orthographic_projection.far = 1.0;
    camera.orthographic_projection.scale = 0.005;
    camera.transform = Transform::from_xyz(0.0, 1.0, 0.0);
    commands.spawn_bundle(camera);

    // This parent entity will receive the model scene as a child entity and the animation components.
    let id = commands.spawn().id();

    // Spawn the first scene in `models/SimpleSkin/SimpleSkin.gltf`
    commands.spawn_scene(asset_server.load::<Scene, _>("models/SimpleSkin/SimpleSkin.gltf#Scene0"));

    // Request animation activation. Once the Gltf is loaded, an animation controller will be created and added to the entity in place of the ActivateGltfAnimation component.
    let gltf_handle = asset_server.load::<Gltf, _>("models/SimpleSkin/SimpleSkin.gltf");
    commands
        .entity(id)
        .insert(ActivateGltfAnimation(gltf_handle.clone()));
}

struct ActivateGltfAnimation(Handle<Gltf>);

/// Replaces ActivateGltfAnimation components with GltfAnimationController when the corresponding Gltf is ready.
fn activate_animations(
    mut commands: Commands,
    query: Query<(Entity, &ActivateGltfAnimation)>,
    gltf_assets: Res<Assets<Gltf>>,
    anim_assets: Res<Assets<GltfAnimation>>,
) {
    for (id, anim_activation) in query.iter() {
        let gltf_handle = &anim_activation.0.clone();
        if !gltf_assets.contains(gltf_handle) {
            continue;
        }

        let gltf = gltf_assets.get(gltf_handle).unwrap();

        commands
            .entity(id)
            .remove::<ActivateGltfAnimation>()
            .insert(GltfAnimationController {
                animations: gltf.animations.to_vec(),
                times: gltf.animations.iter().map(|_| 0.).collect(),
                weights: gltf
                    .animations
                    .iter()
                    .enumerate()
                    // The first animation index begins with weight 1.
                    .map(|(i, _)| if i == 0 { 1. } else { 0. })
                    .collect(),
                start_times: gltf
                    .animations
                    .iter()
                    .map(|a| anim_assets.get(a).unwrap().start_time())
                    .collect(),
                durations: gltf
                    .animations
                    .iter()
                    .map(|a| anim_assets.get(a).unwrap().duration())
                    .collect(),
            });
    }
}

/// Updates the animation controller of the Gltf animation demo scene by setting the weight of the first animation index to one, and updating the time value of all animations.
fn update_animation_controllers(mut query: Query<&mut GltfAnimationController>, time: Res<Time>) {
    for mut ctrl in query.iter_mut() {
        // Here we assume an animation should simply be playing on loop.
        let time_secs = time.seconds_since_startup() as f32;
        for i in 0..ctrl.times.len() {
            ctrl.times[i] = ctrl.start_times[i] + (time_secs % ctrl.durations[i]);
        }

        // Select the animation to be active. We could also choose multiple animations or blend their values, but this example only has a single animation.
        ctrl.weights.fill(0.);
        ctrl.weights[0] = 1.;
    }
}

/// Component containing the data necessary to evaluate Gltf animation property data (translation, rotation, scale, and morph target weights) for target GltfNodes at a given time.
///
/// The update_gltf_animations system supports evaluating and blending multiple animations, given a set of animation index weights. All animations are evaluated and contribute to the final evaluated node property values unless their weight is zero.
#[derive(Debug)]
struct GltfAnimationController {
    animations: Vec<Handle<GltfAnimation>>,
    times: Vec<f32>,
    weights: Vec<f32>,
    start_times: Vec<f32>,
    durations: Vec<f32>,
}

/// Contains a single evaluated animation property value.
enum GltfAnimOutputSample {
    Position(Vec3),
    Rotation(Quat),
    Scale(Vec3),
    MorphTargetWeights(f32), // TODO: I think this should actually be Vec<f32>
}

fn update_gltf_animations(
    gltf_assets: Res<Assets<Gltf>>,
    anim_assets: Res<Assets<GltfAnimation>>,
    query_evaluators: Query<&GltfAnimationController>,
    mut query_targets: Query<(&GltfAnimTargetInfo, &mut Transform)>,
) {
    let mut eval_data = HashMap::<Handle<GltfAnimation>, (f32, f32)>::default();
    for eval in query_evaluators.iter() {
        for (anim_idx, anim_handle) in eval.animations.iter().enumerate() {
            eval_data.insert(
                anim_handle.clone(),
                (eval.times[anim_idx], eval.weights[anim_idx]),
            );
        }
    }

    for (target_info, mut xfm) in query_targets.iter_mut() {
        let gltf_handle = &target_info.gltf;
        let gltf = gltf_assets.get(gltf_handle);
        if gltf.is_none() {
            continue;
        }
        let gltf = gltf.unwrap();

        let anim_idcs = &target_info.animation_indices;
        let chan_idcs = &target_info.channel_indices;

        let anim_handles = &gltf.animations;

        let mut accum_pos = Vec::<(Vec3, f32)>::with_capacity(anim_handles.len());
        let mut accum_rot = Vec::<(Quat, f32)>::with_capacity(anim_handles.len());
        let mut accum_scale = Vec::<(Vec3, f32)>::with_capacity(anim_handles.len());

        // Get each channel, its time, and its blend weight.
        let node_animations: Vec<_> = anim_idcs
            .iter()
            .zip(chan_idcs)
            .filter_map(|(anim_idx, chan_idx)| {
                let anim_handle = &anim_handles[*anim_idx];
                if let Some(anim) = &anim_assets.get(anim_handle.clone()) {
                    let channel = &anim.channels[*chan_idx];

                    let input_vals = eval_data.get(&anim_handle.clone());
                    if input_vals.is_none() {
                        None
                    } else {
                        let (input_time, input_weight) = input_vals.unwrap();
                        Some((channel, input_time, input_weight))
                    }
                } else {
                    None
                }
            })
            .collect();

        // Accumulate weighted animated properties.
        for (channel, input_time, input_weight) in node_animations {
            if *input_weight == 0. {
                continue;
            }

            let output_sample = sample_animation_value(&channel.sampler, *input_time);

            match (&channel.target.path, output_sample) {
                (GltfAnimTargetProperty::Position, GltfAnimOutputSample::Position(pos)) => {
                    accum_pos.push((pos, *input_weight))
                }
                (GltfAnimTargetProperty::Rotation, GltfAnimOutputSample::Rotation(rot)) => {
                    accum_rot.push((rot, *input_weight))
                }
                (GltfAnimTargetProperty::Scale, GltfAnimOutputSample::Scale(scale)) => {
                    accum_scale.push((scale, *input_weight))
                }
                (
                    GltfAnimTargetProperty::MorphTargetWeights,
                    GltfAnimOutputSample::MorphTargetWeights(_weights),
                ) => todo!("Morph target weights NYI."),
                (_, _) => panic!("Mismatch between target property and sampler output type."),
            }
        }

        // Compute blends and assign transform values.
        let translation = {
            if accum_pos.len() > 0 {
                let (pos_sum, weight_sum) = accum_pos
                    .iter()
                    .fold((Vec3::ZERO, 0.), |(acc_pos, acc_w), (pos, w)| {
                        (acc_pos + (*pos * *w), acc_w + w)
                    });
                Some(pos_sum / weight_sum)
            } else {
                None
            }
        };
        let rotation = {
            if accum_rot.len() > 0 {
                Some(accum_rot.iter().fold(Quat::IDENTITY, |acc_rot, (rot, w)| {
                    Quat::lerp(Quat::IDENTITY, *rot, *w) * acc_rot
                }))
            } else {
                None
            }
        };
        let scale = {
            if accum_scale.len() > 0 {
                let (scale_sum, weight_sum) = accum_scale
                    .iter()
                    .fold((Vec3::ZERO, 0.), |(acc_scale, acc_w), (scale, w)| {
                        (acc_scale + (*scale * *w), acc_w + w)
                    });
                Some(scale_sum / weight_sum)
            } else {
                None
            }
        };

        if let Some(t) = translation {
            xfm.translation = t;
        }
        if let Some(r) = rotation {
            xfm.rotation = r;
        }
        if let Some(s) = scale {
            xfm.scale = s;
        }
    }
}

fn sample_animation_value(sampler: &GltfAnimSampler, time: f32) -> GltfAnimOutputSample {
    let times = &sampler.input.0;
    let interp = &sampler.interpolation;
    match &sampler.output {
        GltfAnimOutputValues::Translations(vs) => {
            GltfAnimOutputSample::Position(interpolate_vec3(vs, times, time, interp))
        }
        GltfAnimOutputValues::Rotations(qs) => {
            GltfAnimOutputSample::Rotation(interpolate_quat(qs, times, time, interp))
        }
        GltfAnimOutputValues::Scales(vs) => {
            GltfAnimOutputSample::Scale(interpolate_vec3(vs, times, time, interp))
        }
        GltfAnimOutputValues::MorphTargetWeights(_ws) => {
            // todo!("Support morph target weights")
            GltfAnimOutputSample::MorphTargetWeights(0.)
        }
    }
}

fn interpolate_vec3(
    vec3s: &Vec<Vec3>,
    times: &Vec<f32>,
    t: f32,
    interp: &GltfAnimInterpolation,
) -> Vec3 {
    // Find the two keyframe indices to interpolate between.
    let (ti0, ti1) = times
        .iter()
        .enumerate()
        .rfind(|(_, kt)| t > **kt) // First keyframe,
        .and_then(|(i, _)| Some((i, (i + 1)))) // + next keyframe,
        .and_then(|(i0, i1)| Some((i0, i1.min(vec3s.len() - 1)))) // (clamp for large t)
        .unwrap_or((0, 0)); // Or t < all keys, so both t0, t1 are 0.

    if ti0 == ti1 {
        return vec3s[ti0];
    }
    let (t0, t1) = (times[ti0], times[ti1]);
    match interp {
        GltfAnimInterpolation::Linear => Vec3::lerp(vec3s[ti0], vec3s[ti1], (t - t0) / (t1 - t0)),
        GltfAnimInterpolation::Step => {
            todo!()
        }
        GltfAnimInterpolation::CubicSpline => {
            todo!()
        }
    }
}

fn interpolate_quat(
    quats: &Vec<Quat>,
    times: &Vec<f32>,
    t: f32,
    interp: &GltfAnimInterpolation,
) -> Quat {
    // Find the two keyframe indices to interpolate between.
    let (ti0, ti1) = times
        .iter()
        .enumerate()
        .rfind(|(_, kt)| t > **kt) // First keyframe,
        .and_then(|(i, _)| Some((i, (i + 1)))) // + next keyframe,
        .and_then(|(i0, i1)| Some((i0, i1.min(quats.len() - 1)))) // (clamp for large t)
        .unwrap_or((0, 0)); // Or t < all keys, so both t0, t1 are 0.

    if ti0 == ti1 {
        return quats[ti0];
    }
    let (t0, t1) = (times[ti0], times[ti1]);
    match interp {
        GltfAnimInterpolation::Linear => Quat::lerp(quats[ti0], quats[ti1], (t - t0) / (t1 - t0)),
        GltfAnimInterpolation::Step => {
            todo!()
        }
        GltfAnimInterpolation::CubicSpline => {
            todo!()
        }
    }
}
