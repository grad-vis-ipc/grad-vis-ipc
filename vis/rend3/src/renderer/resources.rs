use crate::{
    bind_merge::BindGroupBuilder,
    datatypes::TextureHandle,
    renderer::{camera::Camera, util, util::SamplerType},
    RendererOptions,
};
use wgpu::{
    BindGroup, BindGroupEntry, BindGroupLayout, BindingResource, Device, Sampler, Surface, SwapChain, Texture,
    TextureView,
};

pub struct RendererGlobalResources {
    pub swapchain: SwapChain,

    pub color_texture: Texture,
    pub color_texture_view: TextureView,
    pub normal_texture: Texture,
    pub normal_texture_view: TextureView,
    pub depth_texture: Texture,
    pub depth_texture_view: TextureView,
    pub color_bg: BindGroup,

    pub camera: Camera,
    pub background_texture: Option<TextureHandle>,

    pub blit_bgl: BindGroupLayout,
    pub prefix_sum_bgl: BindGroupLayout,
    pub pre_cull_bgl: BindGroupLayout,
    pub general_bgl: BindGroupLayout,
    pub object_output_bgl: BindGroupLayout,
    pub object_output_noindirect_bgl: BindGroupLayout,
    pub uniform_bgl: BindGroupLayout,

    pub linear_sampler: Sampler,
    pub shadow_sampler: Sampler,
}
impl RendererGlobalResources {
    pub fn new(device: &Device, surface: &Surface, options: &RendererOptions) -> Self {
        let swapchain = util::create_swapchain(device, surface, options.size, options.vsync);

        let (color_texture, color_texture_view) =
            util::create_framebuffer_texture(device, options.size, util::FramebufferTextureKind::Color);
        let (normal_texture, normal_texture_view) =
            util::create_framebuffer_texture(device, options.size, util::FramebufferTextureKind::Normal);
        let (depth_texture, depth_texture_view) =
            util::create_framebuffer_texture(device, options.size, util::FramebufferTextureKind::Depth);

        let camera = Camera::new_projection(options.size.width as f32 / options.size.height as f32);

        let blit_bgl = util::create_blit_bgl(device);
        let prefix_sum_bgl = util::create_prefix_sum_bgl(device);
        let pre_cull_bgl = util::create_pre_cull_bgl(device);
        let general_bgl = util::create_general_bind_group_layout(device);
        let object_output_bgl = util::create_object_output_bgl(device);
        let object_output_noindirect_bgl = util::create_object_output_noindirect_bgl(device);
        let uniform_bgl = util::create_uniform_bgl(device);

        let linear_sampler = util::create_sampler(device, SamplerType::Linear);
        let shadow_sampler = util::create_sampler(device, SamplerType::Shadow);

        let color_bg = util::create_blit_bg(device, &blit_bgl, &color_texture_view, &linear_sampler);

        Self {
            swapchain,
            color_texture,
            color_texture_view,
            normal_texture,
            normal_texture_view,
            depth_texture,
            depth_texture_view,
            color_bg,
            camera,
            background_texture: None,
            blit_bgl,
            prefix_sum_bgl,
            pre_cull_bgl,
            general_bgl,
            object_output_bgl,
            object_output_noindirect_bgl,
            uniform_bgl,
            linear_sampler,
            shadow_sampler,
        }
    }

    pub fn update(
        &mut self,
        device: &Device,
        surface: &Surface,
        old_options: &mut RendererOptions,
        new_options: RendererOptions,
    ) {
        let dirty = determine_dirty(old_options, &new_options);

        if dirty.contains(DirtyResources::SWAPCHAIN) {
            self.swapchain = util::create_swapchain(device, surface, new_options.size, new_options.vsync);
        }
        if dirty.contains(DirtyResources::CAMERA) {
            self.camera
                .set_aspect_ratio(new_options.size.width as f32 / new_options.size.height as f32);
        }
        if dirty.contains(DirtyResources::FRAMEBUFFER) {
            let (color_texture, color_texture_view) =
                util::create_framebuffer_texture(device, new_options.size, util::FramebufferTextureKind::Color);
            let (normal_texture, normal_texture_view) =
                util::create_framebuffer_texture(device, new_options.size, util::FramebufferTextureKind::Normal);
            let (depth_texture, depth_texture_view) =
                util::create_framebuffer_texture(device, new_options.size, util::FramebufferTextureKind::Depth);
            let color_bg = util::create_blit_bg(device, &self.blit_bgl, &color_texture_view, &self.linear_sampler);

            self.color_texture = color_texture;
            self.color_texture_view = color_texture_view;
            self.normal_texture = normal_texture;
            self.normal_texture_view = normal_texture_view;
            self.depth_texture = depth_texture;
            self.depth_texture_view = depth_texture_view;
            self.color_bg = color_bg;
        }

        *old_options = new_options
    }

    pub fn append_to_bgb<'a>(&'a self, general_bgb: &mut BindGroupBuilder<'a>) {
        general_bgb.append(BindGroupEntry {
            binding: 0,
            resource: BindingResource::Sampler(&self.linear_sampler),
        });
        general_bgb.append(BindGroupEntry {
            binding: 0,
            resource: BindingResource::Sampler(&self.shadow_sampler),
        });
    }
}

bitflags::bitflags! {
    struct DirtyResources: u8 {
        const SWAPCHAIN = 0x01;
        const CAMERA = 0x02;
        const FRAMEBUFFER = 0x04;
    }
}

fn determine_dirty(current: &RendererOptions, new: &RendererOptions) -> DirtyResources {
    let mut dirty = DirtyResources::empty();

    if current.size != new.size {
        dirty |= DirtyResources::SWAPCHAIN;
        dirty |= DirtyResources::CAMERA;
        dirty |= DirtyResources::FRAMEBUFFER;
    }

    if current.vsync != new.vsync {
        dirty |= DirtyResources::SWAPCHAIN;
    }

    dirty
}
