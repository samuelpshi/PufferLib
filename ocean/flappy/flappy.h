/* Flappy: a sample single-agent grid env.
 * Use this as a tutorial and template for your first env.
 * See the Target env for a slightly more complex example.
 * Star PufferLib on GitHub to support. It really, really helps!
 */

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

#define MIN_GAP_Y (GAP_HEIGHT/2 + GAP_HEIGHT)
#define MAX_GAP_Y (SCREEN_HEIGHT - GAP_HEIGHT/2 - GAP_HEIGHT)

// Required struct. Only use floats!
typedef struct {
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    // Any extra fields you add here may be exported in binding.c
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
    float vy;       // bird vertical velocity
    float ep_return;

    float pipe_x[3];     // x position of each pipe
    float pipe_gap_y[3];  // y-center (or top) of the gap for each pipe

    unsigned int rng;
} Flappy;

void add_log(Flappy* env) {
    env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
    env->log.score += env->rewards[0];
    env->log.episode_length += env->tick;
    env->log.episode_return += env->ep_return;
    env->log.n++;
    
}

// Required function
void c_reset(Flappy* env) {
    env->tick = 0;
    
    
    env->y = SCREEN_HEIGHT/2;
    env->vy = 0.0f;
    env->ep_return = 0.0f;

    env->pipe_x[0] = PIPE_LEAD_IN;
    env->pipe_x[1] = env->pipe_x[0] + PIPE_SPACING;
    env->pipe_x[2] = env->pipe_x[1] + PIPE_SPACING;

    for (int i = 0; i < 3; i++) {
        env->pipe_gap_y[i] = MIN_GAP_Y + (rand_r(&env->rng) % (int)(MAX_GAP_Y - MIN_GAP_Y));
    }

    env->observations[0] = env->y / SCREEN_HEIGHT;
    env->observations[1] = env->vy;
    env->observations[2] = env->pipe_x[0] / SCREEN_WIDTH;
    env->observations[3] = env->pipe_gap_y[0] / SCREEN_HEIGHT;
    env->observations[4] = env->pipe_x[1] / SCREEN_WIDTH;
    env->observations[5] = env->pipe_gap_y[1] / SCREEN_HEIGHT;

}

// Required function
void c_step(Flappy* env) {
    env->tick += 1;
    int action = (int)env->actions[0];
    env->terminals[0] = 0;
    env->rewards[0] = 0.1;

    if (action == FLAP) {
        env->vy = FLAP_IMPULSE;  // negative constant
    } else {
        env->vy += GRAVITY;  // positive constant
    }

    env->y += env->vy;

    for (int i = 0; i < 3; i++) {
        env->pipe_x[i] -= SCROLL_SPEED;
    }

    if (env->pipe_x[0] + PIPE_WIDTH < BIRD_X) {
        env->rewards[0] = 1.0;

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


    env->ep_return += env->rewards[0];

    if (hit_pipe || hit_boundary) {
        env->terminals[0] = 1;
        env->rewards[0] = -1.0;
        env->ep_return += env->rewards[0];
        add_log(env);
        c_reset(env);
        return;
    }
    
    env->observations[0] = env->y / SCREEN_HEIGHT;
    env->observations[1] = env->vy;
    env->observations[2] = env->pipe_x[0] / SCREEN_WIDTH;
    env->observations[3] = env->pipe_gap_y[0] / SCREEN_HEIGHT;
    env->observations[4] = env->pipe_x[1] / SCREEN_WIDTH;
    env->observations[5] = env->pipe_gap_y[1] / SCREEN_HEIGHT;

}

// Required function. Should handle creating the client on first call
void c_render(Flappy* env) {
    if (!IsWindowReady()) {
        InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "PufferLib Flappy");
        SetTargetFPS(30);  // or 60, pick one
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});  // background, fine as-is

    // Draw bird as a rectangle centered on (BIRD_X, env->y)
    DrawRectangle(
        BIRD_X - BIRD_WIDTH/2,        // top-left x
        (int)(env->y) - BIRD_HEIGHT/2, // top-left y
        BIRD_WIDTH,
        BIRD_HEIGHT,
        (Color){0, 187, 187, 255}      // your color, can change
    );

    // Draw all 3 pipes — top and bottom rectangles per pipe
    for (int i = 0; i < 3; i++) {
        float gap_top = env->pipe_gap_y[i] - GAP_HEIGHT/2;
        float gap_bottom = env->pipe_gap_y[i] + GAP_HEIGHT/2;

        // Top pipe: from y=0 down to gap_top
        DrawRectangle(
            (int)env->pipe_x[i], 0,
            PIPE_WIDTH, (int)gap_top,
            (Color){0, 187, 0, 255}  // green-ish
        );

        // Bottom pipe: from gap_bottom down to SCREEN_HEIGHT
        DrawRectangle(
            (int)env->pipe_x[i], (int)gap_bottom,
            PIPE_WIDTH, SCREEN_HEIGHT - (int)gap_bottom,
            (Color){0, 187, 0, 255}
        );
    }

    EndDrawing();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(Flappy* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}