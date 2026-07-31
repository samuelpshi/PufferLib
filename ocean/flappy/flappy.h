/* Flappy: a side-scrolling single-agent env. */

#include <stdlib.h>
#include <string.h>
#include "raylib.h"
#include <stdio.h>

#define SCREEN_WIDTH 430
#define SCREEN_HEIGHT 512
#define BIRD_X 80
#define BIRD_WIDTH 30
#define BIRD_HEIGHT 30
#define PIPE_WIDTH 52
#define PIPE_SPACING 180
#define GAP_HEIGHT 125
#define PIPE_LEAD_IN 300
#define SCROLL_SPEED 2.5
#define GRAVITY 0.8f
#define FLAP_IMPULSE -8.5f
#define FLAP 1

// Max fall speed the bird can reach falling the full screen height from rest:
// v = sqrt(2 * GRAVITY * SCREEN_HEIGHT) ~= 28.6. Round up for headroom.
#define MAX_VY 30.0f

// Hard episode cap. Difficulty never ramps in this env (unlike e.g. dino),
// so a sufficiently good policy can survive indefinitely — without a cap,
// add_log would eventually stop firing and every metric would go stale.
#define MAX_TICKS 5000

// Target pipe count used to normalize perf into [0, 1]. ~500 gives a strong
// agent (~460 baseline) headroom above 0.9 without flattening the objective.
#define PERF_TARGET 500.0f

#define MIN_GAP_Y (GAP_HEIGHT/2 + GAP_HEIGHT)
#define MAX_GAP_Y (SCREEN_HEIGHT - GAP_HEIGHT/2 - GAP_HEIGHT)

// Required struct. Only use floats!
typedef struct {
    float perf; // 0-1 normalized: pipes_passed / PERF_TARGET, clamped
    float score; // unnormalized: raw pipes passed
    float episode_return; // sum of agent rewards over episode
    float episode_length; // number of steps of agent episode
    float n; // Required as the last field
} Log;

// Required that you have some struct for your env
typedef struct {
    Log log; // Required field. Env binding code uses this to aggregate logs
    float* observations; // Required. You can use any obs type, but make sure it matches in Python!
    float* actions; // Required
    float* rewards; // Required
    float* terminals; // Required
    int num_agents;
    int tick;

    float y;       // bird vertical position
    float vy;      // bird vertical velocity
    float ep_return;
    float pipes_passed;

    float pipe_x[3];      // x position of each pipe
    float pipe_gap_y[3];  // y-center (or top) of the gap for each pipe

    unsigned int rng;
} Flappy;

void add_log(Flappy* env) {
    env->log.score += env->pipes_passed;
    env->log.perf += fminf(env->pipes_passed / PERF_TARGET, 1.0f);
    env->log.episode_length += env->tick;
    env->log.episode_return += env->ep_return;
    env->log.n++;
}

void compute_observations(Flappy* env) {
    env->observations[0] = env->y / SCREEN_HEIGHT;
    env->observations[1] = env->vy / MAX_VY;
    env->observations[2] = env->pipe_x[0] / SCREEN_WIDTH;
    env->observations[3] = env->pipe_gap_y[0] / SCREEN_HEIGHT;
    env->observations[4] = env->pipe_x[1] / SCREEN_WIDTH;
    env->observations[5] = env->pipe_gap_y[1] / SCREEN_HEIGHT;
}

// Required function
void c_reset(Flappy* env) {
    env->tick = 0;

    env->y = SCREEN_HEIGHT/2;
    env->vy = 0.0f;
    env->ep_return = 0.0f;
    env->pipes_passed = 0.0f;

    env->pipe_x[0] = PIPE_LEAD_IN;
    env->pipe_x[1] = env->pipe_x[0] + PIPE_SPACING;
    env->pipe_x[2] = env->pipe_x[1] + PIPE_SPACING;

    for (int i = 0; i < 3; i++) {
        env->pipe_gap_y[i] = MIN_GAP_Y + (rand_r(&env->rng) % (int)(MAX_GAP_Y - MIN_GAP_Y));
    }

    compute_observations(env);
}

// Required function
void c_step(Flappy* env) {
    env->tick += 1;
    int action = (int)env->actions[0];
    env->terminals[0] = 0;
    env->rewards[0] = 0.01;

    if (action == FLAP) {
        env->vy = FLAP_IMPULSE;  // negative constant
    } else {
        env->vy += GRAVITY;  // positive constant
    }
    if (env->vy > MAX_VY) env->vy = MAX_VY;

    env->y += env->vy;

    for (int i = 0; i < 3; i++) {
        env->pipe_x[i] -= SCROLL_SPEED;
    }

    if (env->pipe_x[0] + PIPE_WIDTH < BIRD_X) {
        env->rewards[0] = 1.0;
        env->pipes_passed += 1.0f;

        env->pipe_x[0] = env->pipe_x[1];
        env->pipe_x[1] = env->pipe_x[2];
        env->pipe_x[2] = env->pipe_x[1] + PIPE_SPACING;

        env->pipe_gap_y[0] = env->pipe_gap_y[1];
        env->pipe_gap_y[1] = env->pipe_gap_y[2];
        env->pipe_gap_y[2] = MIN_GAP_Y + (rand_r(&env->rng) % (int)(MAX_GAP_Y - MIN_GAP_Y));
    }

    int in_pipe_x_range = (BIRD_X + BIRD_WIDTH/2 > env->pipe_x[0]) && (BIRD_X - BIRD_WIDTH/2 < env->pipe_x[0] + PIPE_WIDTH);
    float gap_top = env->pipe_gap_y[0] - GAP_HEIGHT/2;
    float gap_bottom = env->pipe_gap_y[0] + GAP_HEIGHT/2;
    int hit_pipe = in_pipe_x_range && ((env->y - BIRD_HEIGHT/2 < gap_top) || (env->y + BIRD_HEIGHT/2 > gap_bottom));
    int hit_boundary = (env->y - BIRD_HEIGHT/2 < 0) || (env->y + BIRD_HEIGHT/2 > SCREEN_HEIGHT);
    int timed_out = env->tick >= MAX_TICKS;

    env->ep_return += env->rewards[0];

    if (hit_pipe || hit_boundary) {
        env->terminals[0] = 1;
        env->rewards[0] = -1.0;
        env->ep_return += env->rewards[0];
        add_log(env);
        c_reset(env);
        return;
    }

    if (timed_out) {
        // Not a failure — no death penalty, just end the episode so logs
        // stay live and GAE segments stay bounded.
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
        return;
    }

    compute_observations(env);
}

// Required function. Should handle creating the client on first call
void c_render(Flappy* env) {
    if (!IsWindowReady()) {
        InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "PufferLib Flappy");
        SetTargetFPS(30);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    if (IsKeyPressed(KEY_TAB)) {
        ToggleFullscreen();
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    // Draw bird as a rectangle centered on (BIRD_X, env->y)
    DrawRectangle(
        BIRD_X - BIRD_WIDTH/2,
        (int)(env->y) - BIRD_HEIGHT/2,
        BIRD_WIDTH,
        BIRD_HEIGHT,
        (Color){0, 187, 187, 255}
    );

    // Draw all 3 pipes — top and bottom rectangles per pipe
    for (int i = 0; i < 3; i++) {
        float gap_top = env->pipe_gap_y[i] - GAP_HEIGHT/2;
        float gap_bottom = env->pipe_gap_y[i] + GAP_HEIGHT/2;

        DrawRectangle(
            (int)env->pipe_x[i], 0,
            PIPE_WIDTH, (int)gap_top,
            (Color){0, 187, 0, 255}
        );

        DrawRectangle(
            (int)env->pipe_x[i], (int)gap_bottom,
            PIPE_WIDTH, SCREEN_HEIGHT - (int)gap_bottom,
            (Color){0, 187, 0, 255}
        );
    }

    DrawText(TextFormat("%.0f", env->pipes_passed), 10, 10, 20, RAYWHITE);

    EndDrawing();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(Flappy* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}