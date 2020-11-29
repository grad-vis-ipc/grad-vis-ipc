use std::sync::Arc;
use wgpu::{BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindingResource, Device};

pub struct BindGroupBuilder<'a> {
    bindings: Vec<BindGroupEntry<'a>>,
    label: Option<String>,
}
impl<'a> BindGroupBuilder<'a> {
    pub fn new(label: Option<String>) -> Self {
        Self {
            label,
            bindings: Vec::with_capacity(16),
        }
    }

    pub fn append(&mut self, resource: BindingResource<'a>) {
        let index = self.bindings.len();
        self.bindings.push(BindGroupEntry {
            binding: index as u32,
            resource,
        });
    }

    pub fn build(self, device: &Device, layout: &BindGroupLayout) -> Arc<BindGroup> {
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: self.label.as_deref(),
            layout,
            entries: &self.bindings,
        });

        Arc::new(bind_group)
    }
}
