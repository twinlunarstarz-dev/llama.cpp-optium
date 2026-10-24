#include "llama.h"
#include "../src/llama-ext.h"
#include "../src/llama-model-loader.h"

#include "arg.h"
#include "common.h"
#include "fit.h"
#include "log.h"

#include <cinttypes>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

// satisfies -Wmissing-declarations
int llama_fit_params(int argc, char ** argv);

int llama_fit_params(int argc, char ** argv) {
    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FIT_PARAMS)) {
        return 1;
    }

    // Resolve --device raw entries into params.devices, same as the server does,
    // so the MoE fit planner sees the real device list instead of defaulting to 0.
    common_params_resolve_devices(params);

    llama_backend_init();
    llama_numa_init(params.numa);

    // MoE tiered placement: run the early solver before mparams is built so the
    // memory estimate reflects the tiering plan instead of the full model.
    // Mirrors common_init_result() in common/common.cpp; without this the probe
    // measures all 93 GB of a tiered MoE and over-estimates VRAM by the size of
    // the RAM and disk tiers, which makes vram_fits_locked() fail forever.
    // Disabled: the loader's precise per-device plan is authoritative now, and
    // it is what the no_alloc probe below measures. The early heuristic would
    // install a coarser override set and suppress that plan.
    if (false && params.moe_ram_mib > 0 && (params.moe_placement.empty() || params.moe_placement == "auto")) {
        int moe_n_layer = 0, moe_n_expert = 0, moe_n_expert_used = 0;
        std::vector<llama_model_tensor_buft_override> moe_overrides;
        const bool moe_ok = llama_moe_solve_early(
                params.model.path.c_str(),
                params.tensor_split,
                0,
                params.moe_ram_mib,
                params.moe_placement.empty() ? nullptr : params.moe_placement.c_str(),
                params.moe_pin_decode,
                params.moe_hot.empty() ? nullptr : params.moe_hot.c_str(),
                moe_n_layer, moe_n_expert, moe_n_expert_used, moe_overrides);
        if (moe_ok && !moe_overrides.empty()) {
            params.tensor_buft_overrides = moe_overrides;
            if (params.tensor_buft_overrides.back().pattern != nullptr) {
                params.tensor_buft_overrides.push_back({ nullptr, nullptr });
            }
            LOG_INF("%s: MoE solver produced %zu tensor overrides (layers=%d experts=%d)\n",
                    __func__, moe_overrides.size(), moe_n_layer, moe_n_expert);
        } else {
            LOG_WRN("%s: MoE solver produced no overrides; estimate will be MoE-blind\n", __func__);
        }
    }

    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);

    if (!params.fit_params_print) {
        const common_params_fit_status status = common_fit_params(params.model.path.c_str(), &mparams, &cparams,
                params.tensor_split, params.tensor_buft_overrides.data(), params.fit_params_target.data(), params.fit_params_min_ctx,
                nullptr,
                params.verbosity >= LOG_LEVEL_DEBUG ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_ERROR);
        if (status != COMMON_PARAMS_FIT_STATUS_SUCCESS) {
            LOG_ERR("%s: failed to fit CLI arguments to free memory, exiting...\n", __func__);
            exit(1);
        }

        LOG_INF("%s: printing fitted CLI arguments to stdout...\n", __func__);
        common_log_flush(common_log_main());
        printf("-c %" PRIu32 " -ngl %" PRIi32, cparams.n_ctx, mparams.n_gpu_layers);

        size_t nd = llama_max_devices();
        while (nd > 1 && mparams.tensor_split[nd - 1] == 0.0f) {
            nd--;
        }
        if (nd > 1) {
            for (size_t id = 0; id < nd; id++) {
                if (id == 0) {
                    printf(" -ts ");
                }
                printf("%s%" PRIu32, id > 0 ? "," : "", uint32_t(mparams.tensor_split[id]));
            }
        }

        const size_t ntbo = llama_max_tensor_buft_overrides();
        bool any_tbo = false;
        for (size_t itbo = 0; itbo < ntbo && mparams.tensor_buft_overrides[itbo].pattern != nullptr; itbo++) {
            if (itbo == 0) {
                printf(" -ot \"");
            }
            printf("%s%s=%s", itbo > 0 ? "," : "", mparams.tensor_buft_overrides[itbo].pattern, ggml_backend_buft_name(mparams.tensor_buft_overrides[itbo].buft));
            any_tbo = true;
        }
        printf("%s\n", any_tbo ? "\"" : "");
    } else {
        LOG_INF("%s: printing estimated memory in MiB to stdout (device, model, context, compute) ...\n", __func__);
        common_log_flush(common_log_main());

        common_fit_print(params.model.path.c_str(), &mparams, &cparams);
    }

    return 0;
}
