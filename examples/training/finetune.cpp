#include "arg.h"
#include "chat.h"
#include "common.h"
#include "json.h"
#include "log.h"
#include "llama.h"

#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267)  // possible loss of data
#endif

static std::vector<llama_token> finetune_read_teacher_jsonl(llama_context * ctx, llama_model * model, const std::string & path,
                                                             const std::string & template_override) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open teacher JSONL: " + path);
    }

    auto templates = common_chat_templates_init(model, template_override);
    std::vector<llama_token> tokens;
    std::string line;
    size_t n_records = 0;
    size_t line_number = 0;

    while (std::getline(input, line)) {
        ++line_number;
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue;
        }
        try {
            const auto record = common_json::parse(line);
            if (!record.is_object() || !record.contains("messages") || !record.at("messages").is_array()) {
                throw std::runtime_error("expected an object containing a messages array");
            }
            common_chat_templates_inputs inputs;
            inputs.messages = common_chat_msgs_parse_oaicompat(record.at("messages"));
            if (inputs.messages.empty()) {
                throw std::runtime_error("messages array is empty");
            }
            if (record.contains("tools")) {
                inputs.tools = common_chat_tools_parse_oaicompat(record.at("tools"));
            }
            inputs.add_generation_prompt = false;
            inputs.use_jinja = true;
            const auto rendered = common_chat_templates_apply(templates.get(), inputs);
            const auto record_tokens = common_tokenize(ctx, rendered.prompt, tokens.empty(), true);
            if (record_tokens.empty()) {
                throw std::runtime_error("chat template produced no tokens");
            }
            tokens.insert(tokens.end(), record_tokens.begin(), record_tokens.end());
            ++n_records;
        } catch (const std::exception & e) {
            throw std::runtime_error(path + ":" + std::to_string(line_number) + ": " + e.what());
        }
    }
    if (!input.eof() || n_records == 0) {
        throw std::runtime_error("could not read a nonempty teacher dataset: " + path);
    }
    LOG_INF("teacher JSONL: loaded %zu records and %zu tokens using the student chat template\n", n_records, tokens.size());
    return tokens;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string teacher_jsonl;
    std::vector<char *> filtered_args;
    filtered_args.reserve(argc);
    filtered_args.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--teacher-jsonl") {
            if (!teacher_jsonl.empty() || i + 1 >= argc || argv[i + 1][0] == '-') {
                fprintf(stderr, "error: --teacher-jsonl requires one dataset path\n");
                return 1;
            }
            teacher_jsonl = argv[++i];
        } else if (arg.compare(0, 16, "--teacher-jsonl=") == 0) {
            if (!teacher_jsonl.empty() || arg.size() == 16) {
                fprintf(stderr, "error: --teacher-jsonl requires one dataset path\n");
                return 1;
            }
            teacher_jsonl = arg.substr(16);
        } else {
            filtered_args.push_back(argv[i]);
        }
    }

    common_params params;
    params.escape = false;

    common_init();

    if (!common_params_parse((int) filtered_args.size(), filtered_args.data(), params, LLAMA_EXAMPLE_FINETUNE)) {
        return 1;
    }

    if (params.load_mode != LLAMA_LOAD_MODE_NONE) {
        LOG_INF("%s: forcing load_mode = none to enable writable pointers to the weights\n", __func__);
        params.load_mode = LLAMA_LOAD_MODE_NONE;
    }
    if (params.cache_type_k != GGML_TYPE_F32) {
        LOG_INF("%s: force changing k cache type to f32 due to a lack of f16 support for OUT_PROD\n", __func__);
        params.cache_type_k = GGML_TYPE_F32;
    }
    if (params.cache_type_v != GGML_TYPE_F32) {
        LOG_INF("%s: force changing v cache type to f32 due to a lack of f16 support for OUT_PROD\n", __func__);
        params.cache_type_v = GGML_TYPE_F32;
    }

    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_backend_init();
    llama_numa_init(params.numa);
    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == NULL || ctx == NULL) {
        LOG_ERR("%s: unable to load model or context\n", __func__);
        return 1;
    }

    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    }

    std::vector<llama_token> tokens;
    if (teacher_jsonl.empty()) {
        tokens = common_tokenize(ctx, params.prompt, true);
    } else {
        try {
            tokens = finetune_read_teacher_jsonl(ctx, model, teacher_jsonl, params.chat_template);
        } catch (const std::exception & e) {
            LOG_ERR("teacher dataset error: %s\n", e.what());
            return 1;
        }
    }
    ggml_opt_dataset_t dataset = common_opt_dataset_init(ctx, tokens, llama_n_ctx(ctx) / 2);

    struct lr_opt & lr = params.lr;
    LOG_INF("-optimizer %s -lr0 %.2g -wd %.2g -lr-min %.2g -min-epochs %.2g -epochs %d -period %.2g -val %.2g\n",
            ggml_opt_optimizer_name(params.optimizer), (double) lr.lr0, (double) lr.wd, (double) lr.lr_min, (double) lr.decay_epochs,
            (unsigned) lr.epochs, (double) params.n_batch / params.n_ubatch, (double) params.val_split);

    struct llama_opt_params lopt_params{
        /*n_ctx_train     =*/0,
        /*param_filter    =*/llama_opt_param_filter_all,
        /*param_filter_ud =*/nullptr,
        /*get_opt_pars    =*/common_opt_lr_pars,
        /*get_opt_pars_ud =*/&params.lr,
        /*optimizer_type  =*/params.optimizer,
    };
    llama_opt_init(ctx, model, lopt_params);

    const int64_t idata_split = ggml_opt_dataset_ndata(dataset) * (1.0f - params.val_split);

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_eval  = ggml_opt_result_init();

    for (lr.epoch = 0; lr.epoch < lr.epochs; ++lr.epoch) {
        llama_opt_epoch(ctx, dataset, result_train, result_eval, idata_split,
                        ggml_opt_epoch_callback_progress_bar, ggml_opt_epoch_callback_progress_bar);
        fprintf(stderr, "\n");

        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_eval);
    }
    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_eval);

    if (params.out_file.empty()) {
        params.out_file = "finetuned-model.gguf";
    }
    llama_model_save_to_file(model, params.out_file.c_str());

    llama_backend_free();

    return 0;
}
