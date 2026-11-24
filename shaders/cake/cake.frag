#version 450

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec2 screen_pos;
layout(location = 2) in vec2 left_right;
layout(location = 3) flat in int ticks_height;
layout(location = 4) flat in int ticks_start;
layout(location = 5) flat in int buffer_index;
layout(location = 6) flat in int border_width;

layout(location = 0) out vec4 fsout_Color;

layout(push_constant) uniform PushConstants {
    int start_time;
    int end_time;
    int screen_width;
    int screen_height;
    uint skin_id;
} consts;

layout(set = 0, binding = 0) readonly buffer BufferArray
{
    ivec4 BinTree[];
} buffers[256];

const float pi = 3.1415926535897;

ivec4 getNoteAt(int time) {
    int nextIndex = buffers[buffer_index].BinTree[0].x;

    int steps = 0;
    while(steps < 100) {
        ivec4 node = buffers[buffer_index].BinTree[nextIndex];

        int offset = 0;
        if(time < node.x) offset = node.y;
        else offset = node.z;

        if (offset > 0) {
            nextIndex -= offset;
            break;
        }
        nextIndex += offset;

        steps++;
    }

    ivec4 note = buffers[buffer_index].BinTree[nextIndex];

    return note;
}

float ticks_to_screen_y(int ticks) {
    float screen_y = float(ticks - consts.start_time) / float(consts.end_time - consts.start_time);
    return screen_y;
}

void main()
{
    int time = ticks_start + int(ticks_height * v_uv.y);

    ivec4 note = getNoteAt(time);

    vec3 frag_color;

    if (note.z == -1) {
        discard;
    } else {
        frag_color = vec3(((note.z >> 16) & 0xFF) / 255.0, ((note.z >> 8) & 0xFF) / 255.0, (note.z & 0xFF) / 255.0);
    }

    float note_top = ticks_to_screen_y(note.x);
    float note_bottom = ticks_to_screen_y(note.y);

    float y = 1.0 - screen_pos.y;
    float note_top_dist = (y - note_top);
    float note_bottom_dist = (note_bottom - y);

    float note_left_dist = (screen_pos.x - left_right.x);
    float note_right_dist = (left_right.y - screen_pos.x);

    float dx_l = note_left_dist * float(consts.screen_width);
    float dx_r = note_right_dist * float(consts.screen_width);
    float dy_b = note_top_dist * float(consts.screen_height);
    float dy_t = note_bottom_dist * float(consts.screen_height);

    float min_x_dist = min(dx_l, dx_r);
    float min_y_dist = min(dy_b, dy_t);
    float min_dist = min(min_x_dist, min_y_dist);

    float px_w = max((left_right.y - left_right.x) * float(consts.screen_width), 1.0);
    float px_h = max((note_bottom - note_top) * float(consts.screen_height), 1.0);
    float u = clamp(v_uv.x, 0.0, 1.0);
    float v = clamp(note_top_dist / max(note_bottom - note_top, 1e-6), 0.0, 1.0);
    float bw = float(border_width);

    vec3 color = frag_color;

    if (consts.skin_id == 1u) {
        // Wynthesia: rounded glossy note capsules
        vec2 half_sz = vec2(px_w, px_h) * 0.5;
        vec2 p = (vec2(u, v) - 0.5) * vec2(px_w, px_h);
        float r = min(min(half_sz.x, half_sz.y), 5.0);
        vec2 q = abs(p) - (half_sz - vec2(r));
        float sdf = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
        if (r > 1.5 && sdf > 0.0) {
            discard;
        }
        float gloss = 0.72 + 0.34 * sin(pi * clamp(u * 1.12, 0.0, 1.0));
        float spec = exp(-pow((u - 0.27) / 0.16, 2.0)) * 0.30;
        float cap_sheen = (1.0 - smoothstep(0.0, 0.14, min(v, 1.0 - v))) * 0.08;
        color = clamp(frag_color * gloss + vec3(spec + cap_sheen), 0.0, 1.0);
        float rim = max(bw, 1.0);
        if (sdf > -rim) {
            color = frag_color * 0.24;
        }
    } else if (consts.skin_id == 2u) {
        // WFA(+): crisp 3D-beveled rectangular notes
        color = frag_color * mix(1.06, 0.86, u);
        float bevel = bw + 2.5;
        if (min_dist < bw) {
            color = frag_color * 0.12;
        } else if (min(px_w, px_h) > 6.0) {
            if (dy_t < bevel || dx_l < bevel) {
                color = mix(color, vec3(1.0), 0.28);
            } else if (dy_b < bevel || dx_r < bevel) {
                color *= 0.62;
            }
        }
    } else if (consts.skin_id == 3u) {
        // Zanith: sleek luminous modern notes with leading-edge glow
        float h_shade = 0.86 + 0.18 * cos(pi * (u - 0.35));
        float hit_glow = exp(-v * 4.5) * 0.28;
        color = clamp(frag_color * h_shade + mix(frag_color, vec3(1.0), 0.55) * hit_glow, 0.0, 1.0);
        if (min_dist < bw) {
            color = frag_color * 0.15;
        } else if (min_dist < bw + 1.2 && min(px_w, px_h) > 6.0) {
            color = mix(color, vec3(1.0), 0.22);
        }
    } else {
        // Wasaby: original classic cosine shading + crisp dark outline
        color *= (1.0 + cos(pi * 0.5 * v_uv.x)) * 0.5;
        if (min_dist < bw) {
            color = frag_color * 0.2;
        }
    }

    // Square for SRGB
    color = clamp(color, 0.0, 1.0);
    color *= color;
    fsout_Color = vec4(color, 1.0);
}
