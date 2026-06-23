/* Flappy: a sample single-agent grid env.
 * Use this as a tutorial and template for your first env.
 * See the Target env for a slightly more complex example.
 * Star PufferLib on GitHub to support. It really, really helps!
 */

#include <stdlib.h>
#include <string.h>
#include "raylib.h"

const unsigned char DOWN = 0;
const unsigned char UP = 1;

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
    unsigned char y;
    unsigned int rng;
} Flappy;

void add_log(Flappy* env) {
    env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
    env->log.score += env->rewards[0];
    env->log.episode_length += env->tick;
    env->log.episode_return += env->rewards[0];
    env->log.n++;
}

// Required function
void c_reset(Flappy* env) {
    env->tick = 0;
    env->y = rand_r(&env->rng) % 2; // random start: 0 or 1
    env->observations[0] = (float)env->y;

}

// Required function
void c_step(Flappy* env) {
    env->tick += 1;
    int action = (int)env->actions[0];
    env->terminals[0] = 0;
    env->rewards[0] = 0;

    if (action == UP) {
        env->y += 1;
    } else {
        if (env->y > 0) {
            env->y -= 1;
        }
    }

    if (env->y > 1) {
        env->terminals[0] = 1;
        env->rewards[0] = -1.0;

        add_log(env);
        c_reset(env);
        return;
    }
    if (env->tick > 10) {
        env->terminals[0] = 1;
        env->rewards[0] = 0.0;

        add_log(env);
        c_reset(env);
        return;
    }
    
    env->observations[0] = (float)env->y;
}

// Required function. Should handle creating the client on first call
void c_render(Flappy* env) {
    if (!IsWindowReady()) {
        InitWindow(64*1, 64*2, "PufferLib Flappy");
        SetTargetFPS(5);
    }

    // Standard across our envs so exiting is always the same
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    int py = (1 - env->y) * 64;
    DrawRectangle(0, py, 64, 64, (Color){0, 187, 187, 255});
    EndDrawing();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(Flappy* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}