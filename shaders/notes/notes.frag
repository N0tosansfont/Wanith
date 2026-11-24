#version 450

layout(location = 0) in vec3 frag_color;
layout(location = 1) in vec2 frag_tex_coord;
layout(location = 2) in vec2 v_note_size;
layout(location = 3) in vec2 win_size;
layout(location = 4) in flat uint border_width;
layout(location = 5) in flat uint skin_id;

layout(location = 0) out vec4 out_color;

const float pi = 3.1415926535897;

void main() {
    vec2 v_uv = frag_tex_coord;
    vec3 color = frag_color;

    float px_w = max(v_note_size.x * 0.5 * win_size.x, 1.0);
    float px_h = max(v_note_size.y * 0.5 * win_size.y, 1.0);
    float bw = float(border_width);

    float dx_l = v_uv.x * px_w;
    float dx_r = (1.0 - v_uv.x) * px_w;
    float dy_b = v_uv.y * px_h;
    float dy_t = (1.0 - v_uv.y) * px_h;
    float min_edge = min(min(dx_l, dx_r), min(dy_b, dy_t));

    if (skin_id == 1u) {
        // Wynthesia: rounded glossy note capsules
        vec2 half_sz = vec2(px_w, px_h) * 0.5;
        vec2 p = (v_uv - 0.5) * vec2(px_w, px_h);
        float r = min(min(half_sz.x, half_sz.y), 5.0);
        vec2 q = abs(p) - (half_sz - vec2(r));
        float sdf = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
        if (r > 1.5 && sdf > 0.0) {
            discard;
        }
        float gloss = 0.72 + 0.34 * sin(pi * clamp(v_uv.x * 1.12, 0.0, 1.0));
        float spec = exp(-pow((v_uv.x - 0.27) / 0.16, 2.0)) * 0.30;
        float cap_sheen = (1.0 - smoothstep(0.0, 0.14, min(v_uv.y, 1.0 - v_uv.y))) * 0.08;
        color = clamp(frag_color * gloss + vec3(spec + cap_sheen), 0.0, 1.0);
        float rim = max(bw, 1.0);
        if (sdf > -rim) {
            color = frag_color * 0.24;
        }
    } else if (skin_id == 2u) {
        // WFA(+): crisp 3D-beveled rectangular notes
        color = frag_color * mix(1.06, 0.86, v_uv.x);
        float bevel = bw + 2.5;
        if (min_edge < bw) {
            color = frag_color * 0.12;
        } else if (min(px_w, px_h) > 6.0) {
            if (dy_t < bevel || dx_l < bevel) {
                color = mix(color, vec3(1.0), 0.28);
            } else if (dy_b < bevel || dx_r < bevel) {
                color *= 0.62;
            }
        }
    } else if (skin_id == 3u) {
        // Zanith: sleek luminous modern notes with leading-edge glow
        float h_shade = 0.86 + 0.18 * cos(pi * (v_uv.x - 0.35));
        float hit_glow = exp(-v_uv.y * 4.5) * 0.28;
        color = clamp(frag_color * h_shade + mix(frag_color, vec3(1.0), 0.55) * hit_glow, 0.0, 1.0);
        if (min_edge < bw) {
            color = frag_color * 0.15;
        } else if (min_edge < bw + 1.2 && min(px_w, px_h) > 6.0) {
            color = mix(color, vec3(1.0), 0.22);
        }
    } else {
        // Wasaby: original classic cosine shading + crisp dark outline
        color *= (1.0 + cos(pi * 0.5 * v_uv.x)) * 0.5;
        if (min_edge < bw) {
            color = frag_color * 0.2;
        }
    }

    // Square for SRGB
    color = clamp(color, 0.0, 1.0);
    color *= color;
    out_color = vec4(color, 1.0);
}
