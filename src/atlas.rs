//! Foam atlas: every foam frame baked into one RGB texture, for a self-contained GLB.
//!
//! Frames sit in a grid. Each cell is the `M`x`M` foam image surrounded by a `GUTTER`-texel border
//! filled with *wrapped* copies of the image (the foam is periodic), so bilinear filtering at a
//! cell edge reads the correct neighbours instead of the next frame. A per-frame
//! `KHR_texture_transform` (offset + scale) selects the cell.
//!
//! The texture holds final base colour, `lerp(water, foam, F)` in linear light, sRGB-encoded.

use anyhow::{ensure, Result};

/// Wrapped border (texels) around each cell; 2 covers bilinear filtering plus a little slack.
pub const GUTTER: usize = 2;
/// Largest atlas side we will emit (the common GPU texture limit).
pub const MAX_ATLAS_SIDE: usize = 16384;

/// Placement of `frames` cells of `resolution`² texels in one atlas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtlasLayout {
    pub frames: usize,
    pub resolution: usize,
    pub cols: usize,
    pub rows: usize,
}

impl AtlasLayout {
    /// # Errors
    /// If there are no frames, the resolution is zero, or the atlas would exceed
    /// [`MAX_ATLAS_SIDE`] (reduce `--foam-resolution` or the frame count).
    pub fn new(frames: usize, resolution: usize) -> Result<Self> {
        ensure!(frames >= 1, "atlas needs at least one frame");
        ensure!(resolution >= 1, "atlas resolution must be >= 1");
        let cols = (frames as f64).sqrt().ceil() as usize;
        let rows = frames.div_ceil(cols);
        let layout = Self { frames, resolution, cols, rows };
        ensure!(
            layout.width() <= MAX_ATLAS_SIDE && layout.height() <= MAX_ATLAS_SIDE,
            "foam atlas would be {}x{} px for {frames} frames of {resolution}^2 (limit {MAX_ATLAS_SIDE}); \
             lower --foam-resolution or shorten/slow the animation",
            layout.width(),
            layout.height()
        );
        Ok(layout)
    }

    fn cell(&self) -> usize {
        self.resolution + 2 * GUTTER
    }
    pub fn width(&self) -> usize {
        self.cols * self.cell()
    }
    pub fn height(&self) -> usize {
        self.rows * self.cell()
    }

    /// `KHR_texture_transform` `(offset, scale)` that maps UV (0..1) onto frame `frame`'s image
    /// (excluding the gutter). Image row 0 is the top (glTF v = 0), as in the foam PNGs.
    pub fn uv_transform(&self, frame: usize) -> ([f32; 2], [f32; 2]) {
        debug_assert!(frame < self.frames);
        let (col, row) = (frame % self.cols, frame / self.cols);
        let (w, h) = (self.width() as f64, self.height() as f64);
        let m = self.resolution as f64;
        let ox = (col * self.cell() + GUTTER) as f64 / w;
        let oy = (row * self.cell() + GUTTER) as f64 / h;
        ([ox as f32, oy as f32], [(m / w) as f32, (m / h) as f32])
    }
}

fn srgb_encode(linear: f32) -> u8 {
    let c = linear.clamp(0.0, 1.0);
    let s = if c <= 0.003_130_8 { 12.92 * c } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 };
    (s * 255.0).round() as u8
}

/// Render the atlas as RGB8 (`width * height * 3` bytes, row 0 at the top).
///
/// `water` and `foam` are linear RGB; each texel is `lerp(water, foam, F)`.
///
/// # Errors
/// If `frames` does not match the layout, or a mask has the wrong length or a non-finite value.
pub fn render_atlas(layout: &AtlasLayout, frames: &[Vec<f32>], water: [f32; 3], foam: [f32; 3]) -> Result<Vec<u8>> {
    let m = layout.resolution;
    ensure!(frames.len() == layout.frames, "{} frames given for a layout of {}", frames.len(), layout.frames);
    for (i, f) in frames.iter().enumerate() {
        ensure!(f.len() == m * m, "frame {i} has {} values, expected {}", f.len(), m * m);
        ensure!(f.iter().all(|v| v.is_finite()), "frame {i} contains a non-finite value");
    }

    let (w, h, cell) = (layout.width(), layout.height(), layout.cell());
    // Unused cells (after the last frame) repeat the water colour.
    let water_px = [srgb_encode(water[0]), srgb_encode(water[1]), srgb_encode(water[2])];
    let mut out = Vec::with_capacity(w * h * 3);
    for _ in 0..w * h {
        out.extend_from_slice(&water_px);
    }

    // Colour per quantised foam value: avoids a powf per texel (256 levels is enough for 8-bit).
    let lut: Vec<[u8; 3]> = (0..=255u32)
        .map(|q| {
            let f = q as f32 / 255.0;
            let mix = |i: usize| srgb_encode(water[i] + (foam[i] - water[i]) * f);
            [mix(0), mix(1), mix(2)]
        })
        .collect();

    for (n, mask) in frames.iter().enumerate() {
        let (col, row) = (n % layout.cols, n / layout.cols);
        for cy in 0..cell {
            // gutter rows/cols wrap around the periodic image
            let sy = (cy as isize - GUTTER as isize).rem_euclid(m as isize) as usize;
            for cx in 0..cell {
                let sx = (cx as isize - GUTTER as isize).rem_euclid(m as isize) as usize;
                let q = (mask[sy * m + sx].clamp(0.0, 1.0) * 255.0).round() as usize;
                let px = ((row * cell + cy) * w + col * cell + cx) * 3;
                out[px..px + 3].copy_from_slice(&lut[q]);
            }
        }
    }
    Ok(out)
}

/// Encode RGB8 pixels as a PNG.
pub fn encode_rgb_png(pixels: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
    ensure!(pixels.len() == width * height * 3, "pixel buffer size does not match {width}x{height}");
    let mut out = Vec::new();
    let mut encoder = png::Encoder::new(&mut out, width as u32, height as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(pixels)?;
    writer.finish()?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const WATER: [f32; 3] = [0.0, 0.3, 0.8];
    const FOAM: [f32; 3] = [0.9, 0.9, 0.9];

    fn px(atlas: &[u8], layout: &AtlasLayout, x: usize, y: usize) -> [u8; 3] {
        let i = (y * layout.width() + x) * 3;
        [atlas[i], atlas[i + 1], atlas[i + 2]]
    }

    /// Apply a KHR_texture_transform to a UV and return atlas pixel coordinates.
    fn atlas_texel(layout: &AtlasLayout, frame: usize, u: f32, v: f32) -> (usize, usize) {
        let (o, s) = layout.uv_transform(frame);
        (
            ((o[0] + u * s[0]) * layout.width() as f32).floor() as usize,
            ((o[1] + v * s[1]) * layout.height() as f32).floor() as usize,
        )
    }

    #[test]
    fn layout_is_near_square_and_checks_limits() {
        let l = AtlasLayout::new(97, 256).unwrap();
        assert_eq!((l.cols, l.rows), (10, 10));
        assert_eq!(l.width(), 10 * 260);
        assert_eq!(AtlasLayout::new(1, 8).unwrap().cols, 1);
        assert!(AtlasLayout::new(0, 8).is_err());
        assert!(AtlasLayout::new(10, 0).is_err());
        let err = AtlasLayout::new(2000, 512).unwrap_err().to_string();
        assert!(err.contains("--foam-resolution"), "{err}");
    }

    #[test]
    fn transform_maps_uv_corners_to_the_cell_interior() {
        let l = AtlasLayout::new(5, 8).unwrap(); // 3 cols x 2 rows
        for f in 0..5 {
            let (col, row) = (f % l.cols, f / l.cols);
            let (x0, y0) = atlas_texel(&l, f, 0.0, 0.0);
            assert_eq!((x0, y0), (col * 12 + GUTTER, row * 12 + GUTTER));
            // last texel centre of the cell (u = v = 1 - half texel)
            let e = 1.0 - 0.5 / 8.0;
            let (x1, y1) = atlas_texel(&l, f, e, e);
            assert_eq!((x1, y1), (col * 12 + GUTTER + 7, row * 12 + GUTTER + 7));
        }
    }

    #[test]
    fn pixels_land_where_the_transform_says_with_row_zero_on_top() {
        let m = 8;
        let l = AtlasLayout::new(3, m).unwrap();
        let mut frames = vec![vec![0.0f32; m * m]; 3];
        frames[1][2 * m + 5] = 1.0; // frame 1: col 5, row 2
        let atlas = render_atlas(&l, &frames, WATER, FOAM).unwrap();
        // texel centre in UV: ((5+.5)/8, (2+.5)/8)
        let (x, y) = atlas_texel(&l, 1, 5.5 / 8.0, 2.5 / 8.0);
        assert_eq!(px(&atlas, &l, x, y), [srgb_encode(0.9); 3]);
        // the same texel in other frames is water
        let (x, y) = atlas_texel(&l, 0, 5.5 / 8.0, 2.5 / 8.0);
        assert_eq!(px(&atlas, &l, x, y), [0, srgb_encode(0.3), srgb_encode(0.8)]);
    }

    #[test]
    fn gutter_holds_wrapped_copies() {
        let m = 8;
        let l = AtlasLayout::new(2, m).unwrap();
        let mut frames = vec![vec![0.0f32; m * m]; 2];
        frames[0][3 * m + (m - 1)] = 1.0; // far-right texel of row 3
        frames[0][(m - 1) * m + 2] = 1.0; // bottom texel of col 2
        let atlas = render_atlas(&l, &frames, WATER, FOAM).unwrap();
        let (x0, y0) = atlas_texel(&l, 0, 0.0, 0.0);
        let foam_px = [srgb_encode(0.9); 3];
        // just left of the image's first column == wrapped far-right texel (row 3)
        assert_eq!(px(&atlas, &l, x0 - 1, y0 + 3), foam_px);
        assert_eq!(px(&atlas, &l, x0 - 2, y0 + 3), [0, srgb_encode(0.3), srgb_encode(0.8)]);
        // just above the first row == wrapped bottom texel (col 2)
        assert_eq!(px(&atlas, &l, x0 + 2, y0 - 1), foam_px);
        // and it does not bleed into the neighbouring cell's interior
        let (x1, _) = atlas_texel(&l, 1, 0.0, 0.0);
        assert_eq!(px(&atlas, &l, x1, y0 + 3), [0, srgb_encode(0.3), srgb_encode(0.8)]);
    }

    #[test]
    fn colours_interpolate_in_linear_light() {
        let l = AtlasLayout::new(1, 2).unwrap();
        let atlas = render_atlas(&l, &[vec![0.0, 0.5, 1.0, 0.25]], WATER, FOAM).unwrap();
        let (x, y) = atlas_texel(&l, 0, 0.75, 0.25); // texel (1, 0) = 0.5
        let mid = px(&atlas, &l, x, y);
        // foam value 0.5 is quantised to 128/255 for the colour lookup
        let f = 128.0 / 255.0;
        assert_eq!(mid[0], srgb_encode(0.9 * f));
        assert_eq!(mid[1], srgb_encode(0.3 + 0.6 * f));
        assert_ne!(mid[1], srgb_encode(0.3 + 0.6 * 0.5 * 0.5)); // not interpolating in sRGB space
        assert_eq!(srgb_encode(0.0), 0);
        assert_eq!(srgb_encode(1.0), 255);
        assert_eq!(srgb_encode(0.5), 188); // well-known sRGB value
    }

    #[test]
    fn rejects_inconsistent_input() {
        let l = AtlasLayout::new(2, 4).unwrap();
        assert!(render_atlas(&l, &[vec![0.0; 16]], WATER, FOAM).is_err()); // wrong frame count
        assert!(render_atlas(&l, &[vec![0.0; 16], vec![0.0; 15]], WATER, FOAM).is_err());
        assert!(render_atlas(&l, &[vec![0.0; 16], vec![f32::NAN; 16]], WATER, FOAM).is_err());
        assert!(encode_rgb_png(&[0; 5], 2, 2).is_err());
    }

    #[test]
    fn png_round_trips() {
        let l = AtlasLayout::new(1, 4).unwrap();
        let mask: Vec<f32> = (0..16).map(|i| i as f32 / 15.0).collect();
        let atlas = render_atlas(&l, &[mask], WATER, FOAM).unwrap();
        let bytes = encode_rgb_png(&atlas, l.width(), l.height()).unwrap();
        let mut reader = png::Decoder::new(&bytes[..]).read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).unwrap();
        assert_eq!((info.width as usize, info.height as usize), (l.width(), l.height()));
        assert_eq!(&buf[..info.buffer_size()], &atlas[..]);
    }
}
