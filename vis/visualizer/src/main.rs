use fnv::FnvBuildHasher;
use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
use imgui::FontSource;
use obj::{IndexTuple, Obj, ObjMaterial};
use pico_args::Arguments;
use rend3::{
    datatypes::{
        AffineTransform, AlbedoComponent, CameraLocation, DirectionalLight, Material, MaterialComponent,
        MaterialHandle, Mesh, MeshHandle, ModelVertex, Object, ObjectHandle, RendererTextureFormat, Texture,
        TextureHandle,
    },
    list::{DefaultPipelines, DefaultShaders},
    Renderer, RendererOptions, VSyncMode,
};
use std::{
    collections::HashMap,
    fs::File,
    hash::BuildHasher,
    io::BufReader,
    sync::Arc,
    time::{Duration, Instant},
};
use switchyard::{threads, Switchyard};
use winit::{
    dpi::PhysicalSize,
    event::{DeviceEvent, ElementState, Event, KeyboardInput, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

mod platform;

fn load_skybox(renderer: &Renderer) {
    let dds = ddsfile::Dds::read(&mut BufReader::new(
        File::open("tmp/skybox/skybox-compressed.dds").unwrap(),
    ))
    .unwrap();

    let handle = renderer.add_texture_cube(Texture {
        format: RendererTextureFormat::Bc7Srgb,
        width: dds.get_width(),
        height: dds.get_height(),
        data: dds.data,
        label: Some("background".into()),
    });
    renderer.set_background_texture(handle);
}

fn load_obj(renderer: &Renderer, file: &str) -> ObjectHandle {
    rend3::span!(obj_guard, INFO, "Loading Obj");

    let mut object = Obj::load(file).unwrap();

    drop(obj_guard);

    rend3::span!(_guard, INFO, "Converting Mesh");

    let mut mesh = Mesh {
        vertices: vec![],
        indices: vec![],
        material_count: 1,
    };

    let mut translation: HashMap<(usize, Option<usize>, Option<usize>), u32> = HashMap::new();
    let mut vert_count = 0_u32;
    for group in &object.data.objects[0].groups {
        for polygon in &group.polys {
            for &IndexTuple(position_idx, texture_idx, normal_idx) in &polygon.0 {
                let vert_idx = if let Some(&vert_idx) = translation.get(&(position_idx, normal_idx, texture_idx)) {
                    vert_idx
                } else {
                    let this_vert = vert_count;
                    vert_count += 1;

                    let position = object.data.position[position_idx];
                    let normal = object.data.normal[normal_idx.unwrap()];
                    let texture_coords = object.data.texture[texture_idx.unwrap()];

                    mesh.vertices.push(ModelVertex {
                        position: Vec3::from(position),
                        normal: Vec3::from(normal),
                        uv: Vec2::from(texture_coords),
                        color: [255; 4],
                    });

                    translation.insert((position_idx, texture_idx, normal_idx), this_vert);

                    this_vert
                };

                mesh.indices.push(vert_idx);
            }
        }
    }

    let mesh = renderer.add_mesh(mesh);

    let material = renderer.add_material(Material {
        albedo: AlbedoComponent::Value(Vec4::new(0.5, 0.5, 0.0, 1.0)),
        ..Material::default()
    });

    renderer.add_object(Object {
        mesh,
        material,
        transform: AffineTransform {
            transform: Mat4::identity(),
        },
    })
}

fn button_pressed<Hash: BuildHasher>(map: &HashMap<u32, bool, Hash>, key: u32) -> bool {
    map.get(&key).map_or(false, |b| *b)
}

fn extract_backend(value: &str) -> Result<wgpu::Backend, &'static str> {
    Ok(match value.to_lowercase().as_str() {
        "vulkan" | "vk" => wgpu::Backend::Vulkan,
        "dx12" | "12" => wgpu::Backend::Dx12,
        "dx11" | "11" => wgpu::Backend::Dx11,
        "metal" | "mtl" => wgpu::Backend::Metal,
        "opengl" | "gl" => wgpu::Backend::Gl,
        _ => return Err("backend requested but not found"),
    })
}

fn extract_mode(value: &str) -> Result<rend3::RendererMode, &'static str> {
    Ok(match value.to_lowercase().as_str() {
        "legacy" | "c" | "cpu" => rend3::RendererMode::CPUPowered,
        "modern" | "g" | "gpu" => rend3::RendererMode::GPUPowered,
        _ => return Err("mode requested but not found"),
    })
}

fn main() {
    wgpu_subscriber::initialize_default_subscriber(None);

    let mut args = Arguments::from_env();
    let desired_backend = args.value_from_fn(["-b", "--backend"], extract_backend).ok();
    let desired_device: Option<String> = args
        .value_from_str(["-d", "--device"])
        .ok()
        .map(|s: String| s.to_lowercase());
    let desired_mode = args.value_from_fn(["-m", "--mode"], extract_mode).ok();

    rend3::span_transfer!(_ -> main_thread_span, INFO, "Main Thread Setup");
    rend3::span_transfer!(_ -> event_loop_span, INFO, "Building Event Loop");

    let event_loop = EventLoop::new();

    rend3::span_transfer!(event_loop_span -> window_span, INFO, "Building Window");

    let window = {
        let mut builder = WindowBuilder::new();
        builder = builder.with_title("scene-viewer");
        builder.build(&event_loop).expect("Could not build window")
    };

    rend3::span_transfer!(window_span -> imgui_span, INFO, "Building imgui");

    let mut imgui = imgui::Context::create();
    let mut platform = imgui_winit_support::WinitPlatform::init(&mut imgui);
    platform.attach_window(imgui.io_mut(), &window, imgui_winit_support::HiDpiMode::Default);
    imgui.set_ini_filename(None);
    imgui.fonts().add_font(&[FontSource::DefaultFontData {
        config: Some(imgui::FontConfig {
            oversample_h: 3,
            oversample_v: 1,
            pixel_snap_h: true,
            size_pixels: 13.0,
            ..imgui::FontConfig::default()
        }),
    }]);

    rend3::span_transfer!(imgui_span -> switchyard_span, INFO, "Building Switchyard");

    let yard = Arc::new(
        Switchyard::new(
            2,
            // threads::single_pool_single_thread(Some("scene-viewer".into()), None),
            threads::double_pool_one_to_one(threads::thread_info(), Some("scene-viewer")),
            || (),
        )
        .unwrap(),
    );

    rend3::span_transfer!(switchyard_span -> renderer_span, INFO, "Building Renderer");

    let mut options = RendererOptions {
        vsync: VSyncMode::Off,
        size: window.inner_size(),
    };

    let renderer = futures::executor::block_on(rend3::Renderer::new(
        &window,
        Arc::clone(&yard),
        &mut imgui,
        desired_backend,
        desired_device,
        desired_mode,
        options.clone(),
    ))
    .unwrap();

    let pipelines = futures::executor::block_on(async {
        let shaders = DefaultShaders::new(&renderer).await;
        DefaultPipelines::new(&renderer, &shaders).await
    });

    rend3::span_transfer!(renderer_span -> loading_span, INFO, "Loading resources");

    let sphere = load_obj(&renderer, "tmp/sphere.obj");
    load_skybox(&renderer);

    renderer.add_directional_light(DirectionalLight {
        color: Vec3::one(),
        intensity: 10.0,
        direction: Vec3::new(-1.0, -1.0, 0.0),
    });

    rend3::span_transfer!(loading_span -> _);
    rend3::span_transfer!(main_thread_span -> _);

    let mut scancode_status = HashMap::with_hasher(FnvBuildHasher::default());

    let mut camera_location = CameraLocation::default();

    let mut timestamp_last_second = Instant::now();
    let mut timestamp_last_frame = Instant::now();
    let mut frames = 0_usize;

    event_loop.run(move |event, _window_target, control| match event {
        Event::MainEventsCleared => {
            frames += 1;
            let now = Instant::now();
            let elapsed_since_second = now - timestamp_last_second;
            if elapsed_since_second > Duration::from_secs(1) {
                println!(
                    "{} frames over {:.3}s: {:.3}ms/frame",
                    frames,
                    elapsed_since_second.as_secs_f32(),
                    elapsed_since_second.as_secs_f64() * 1000.0 / frames as f64
                );
                timestamp_last_second = now;
                frames = 0;
            }
            let delta_time = now - timestamp_last_frame;
            timestamp_last_frame = now;

            let mut input = String::new();
            std::io::stdin().read_line(&mut input).unwrap();
            let input_trimmed = input.trim();

            // let trim_fluff = &input_trimmed[18..input_trimmed.len() - 1];
            let mut split = input_trimmed.split(",").map(str::trim);

            let first: f32 = split.next().unwrap().parse().unwrap();
            let second: f32 = split.next().unwrap().parse().unwrap();
            let third: f32 = split.next().unwrap().parse().unwrap();

            println!("({}, {}, {})", first, second, third);

            renderer.set_object_transform(
                sphere,
                AffineTransform {
                    transform: Mat4::from_translation(Vec3::new(first, second, third)),
                },
            );

            let velocity = if button_pressed(&scancode_status, platform::Scancodes::SHIFT) {
                10.0
            } else {
                1.0
            };
            if button_pressed(&scancode_status, platform::Scancodes::W) {
                camera_location.location.z += velocity * delta_time.as_secs_f32();
            }
            if button_pressed(&scancode_status, platform::Scancodes::S) {
                camera_location.location.z -= velocity * delta_time.as_secs_f32();
            }
            if button_pressed(&scancode_status, platform::Scancodes::A) {
                camera_location.location.x -= velocity * delta_time.as_secs_f32();
            }
            if button_pressed(&scancode_status, platform::Scancodes::D) {
                camera_location.location.x += velocity * delta_time.as_secs_f32();
            }
            if button_pressed(&scancode_status, platform::Scancodes::Q) {
                camera_location.location.y += velocity * delta_time.as_secs_f32();
            }
            if button_pressed(&scancode_status, platform::Scancodes::Z) {
                camera_location.location.y -= velocity * delta_time.as_secs_f32();
            }

            window.request_redraw();
        }
        Event::WindowEvent {
            event:
                WindowEvent::KeyboardInput {
                    input: KeyboardInput { scancode, state, .. },
                    ..
                },
            ..
        } => {
            scancode_status.insert(
                scancode,
                match state {
                    ElementState::Pressed => true,
                    ElementState::Released => false,
                },
            );
        }
        Event::DeviceEvent {
            event:
                DeviceEvent::MouseMotion {
                    delta: (delta_x, delta_y),
                    ..
                },
            ..
        } => {
            const TAU: f32 = std::f32::consts::PI * 2.0;

            camera_location.yaw += (delta_x / 1000.0) as f32;
            camera_location.pitch += (delta_y / 1000.0) as f32;
            if camera_location.yaw < 0.0 {
                camera_location.yaw += TAU;
            } else if camera_location.yaw >= TAU {
                camera_location.yaw -= TAU;
            }
            camera_location.pitch = camera_location
                .pitch
                .max(-std::f32::consts::FRAC_PI_2 + 0.0001)
                .min(std::f32::consts::FRAC_PI_2 - 0.0001);
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(size),
            ..
        } => {
            options.size = size;
        }
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control = ControlFlow::Exit;
        }
        Event::RedrawRequested(_) => {
            rend3::span_transfer!(_ -> redraw_span, INFO, "Redraw");

            renderer.set_camera_location(camera_location);
            renderer.set_options(options.clone());
            let handle = renderer.render(rend3::list::default_render_list(
                renderer.mode(),
                PhysicalSize {
                    width: (options.size.width as f32 * 1.0) as u32,
                    height: (options.size.height as f32 * 1.0) as u32,
                },
                &pipelines,
            ));

            rend3::span_transfer!(redraw_span -> render_wait_span, INFO, "Waiting for render");
            futures::executor::block_on(handle);
        }
        _ => {}
    })
}
