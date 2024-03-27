#include "bert.h"
#include "ggml.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <regex>
#include <thread>
#include <algorithm>
#include <unistd.h>

// default hparams (all-MiniLM-L6-v2)
struct bert_hparams
{
    int32_t n_vocab = 30522;
    int32_t n_max_tokens = 512;
    int32_t n_embd = 256;
    int32_t n_intermediate = 1536;
    int32_t n_head = 12;
    int32_t n_layer = 6;
    int32_t f16 = 1;
};

struct bert_layer
{
    // normalization
    struct ggml_tensor *ln_att_w;
    struct ggml_tensor *ln_att_b;

    struct ggml_tensor *ln_out_w;
    struct ggml_tensor *ln_out_b;

    // attention
    struct ggml_tensor *q_w;
    struct ggml_tensor *q_b;
    struct ggml_tensor *k_w;
    struct ggml_tensor *k_b;
    struct ggml_tensor *v_w;
    struct ggml_tensor *v_b;

    struct ggml_tensor *o_w;
    struct ggml_tensor *o_b;

    // ff
    struct ggml_tensor *ff_i_w;
    struct ggml_tensor *ff_i_b;

    struct ggml_tensor *ff_o_w;
    struct ggml_tensor *ff_o_b;
};

struct bert_vocab
{
    std::map<std::string, bert_vocab_id> token_to_id;
    std::map<std::string, bert_vocab_id> subword_token_to_id;

    std::map<bert_vocab_id, std::string> _id_to_token;
    std::map<bert_vocab_id, std::string> _id_to_subword_token;
};

struct bert_model
{
    bert_hparams hparams;

    // embeddings weights
    struct ggml_tensor *word_embeddings;
    struct ggml_tensor *token_type_embeddings;
    struct ggml_tensor *position_embeddings;
    struct ggml_tensor *ln_e_w;
    struct ggml_tensor *ln_e_b;

    std::vector<bert_layer> layers;

    // classifier weights
    struct ggml_tensor *cls_w;
    struct ggml_tensor *cls_b;

    // pooler (baaaaka baaaka)
    struct ggml_tensor *plr_w;
    struct ggml_tensor *plr_b;

    struct ggml_context *ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
};

// Replacement for std::vector<uint8_t> that doesn't require zero-initialization.
struct bert_buffer {
    uint8_t * data = NULL;
    size_t size = 0;

    void resize(size_t size) {
        delete[] data;
        data = new uint8_t[size];
        this->size = size;
    }

    ~bert_buffer() {
        delete[] data;
    }
};


struct bert_ctx
{
    bert_model model;
    bert_vocab vocab;

    size_t mem_per_token;
    int64_t mem_per_input;
    int32_t max_batch_n;
    bert_buffer buf_compute;
};

int32_t bert_n_embd(bert_ctx * ctx)
{
    return 6;//ctx->model.hparams.n_embd;
}

int32_t bert_n_max_tokens(bert_ctx * ctx)
{
    return ctx->model.hparams.n_max_tokens;
}

const char* bert_vocab_id_to_token(bert_ctx * ctx, bert_vocab_id id) {
    bert_vocab & vocab = ctx->vocab;
    auto it = vocab._id_to_token.find(id);
    if (it != vocab._id_to_token.end())
    {
        return it->second.c_str();
    }
    it = vocab._id_to_subword_token.find(id);
    if (it != vocab._id_to_subword_token.end())
    {
        return it->second.c_str();
    }
    return "[UNK TOKEN from bert_vocab]";
}

//
// Cli interface
//

void bert_print_usage(char **argv, const bert_params &params)
{
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -s SEED, --seed SEED  RNG seed (default: -1)\n");
    fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -p PROMPT, --prompt PROMPT\n");
    fprintf(stderr, "                        prompt to start generation with (default: random)\n");
    fprintf(stderr, "  --port p     port to bind in server mode (default: %d)\n", params.port);
    fprintf(stderr, "  -m FNAME, --model FNAME\n");
    fprintf(stderr, "                        model path (default: %s)\n", params.model);
    fprintf(stderr, "\n");
}


bool bert_params_parse(int argc, char **argv, bert_params &params)
{
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "-t" || arg == "--threads")
        {
            params.n_threads = std::stoi(argv[++i]);
        }
        else if (arg == "-p" || arg == "--prompt")
        {
            params.prompt = argv[++i];
        }
        else if (arg == "--port")
        {
            params.port = std::stoi(argv[++i]);
        }
        else if (arg == "-m" || arg == "--model")
        {
            params.model = argv[++i];
        }
        else if (arg == "-h" || arg == "--help")
        {
            bert_print_usage(argv, params);
            exit(0);
        }
        else
        {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            bert_print_usage(argv, params);
            exit(0);
        }
    }

    return true;
}

//
// Tokenizing
//

static size_t utf8_len(char src)
{
    const size_t lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

std::string stripAccents(const std::string &inputString)
{
    std::string resultString;
    std::map<std::string, char> accentMap = {{"À", 'A'},{"Á", 'A'},
        {"Â", 'A'},{"Ã", 'A'},{"Ä", 'A'},{"Å", 'A'},{"à", 'a'},{"á", 'a'},
        {"â", 'a'},{"ã", 'a'},{"ä", 'a'},{"å", 'a'},{"È", 'E'},{"É", 'E'},
        {"Ê", 'E'},{"Ë", 'E'},{"è", 'e'},{"é", 'e'},{"ê", 'e'},{"ë", 'e'},
        {"Ì", 'I'},{"Í", 'I'},{"Î", 'I'},{"Ï", 'I'},{"ì", 'i'},{"í", 'i'},
        {"î", 'i'},{"ï", 'i'},{"Ò", 'O'},{"Ó", 'O'},{"Ô", 'O'},{"Õ", 'O'},
        {"Ö", 'O'},{"ò", 'o'},{"ó", 'o'},{"ô", 'o'},{"õ", 'o'},{"ö", 'o'},
        {"Ù", 'U'},{"Ú", 'U'},{"Û", 'U'},{"Ü", 'U'},{"ù", 'u'},{"ú", 'u'},
        {"û", 'u'},{"ü", 'u'},{"Ý", 'Y'},{"ý", 'y'},{"Ç", 'C'},{"ç", 'c'},
        {"Ñ", 'N'},{"ñ", 'n'},
    };

    for (size_t i = 0; i < inputString.length();)
    {
        int len = utf8_len(inputString[i]);
        std::string curChar = inputString.substr(i, len);
        auto iter = accentMap.find(curChar);
        if (iter != accentMap.end())
        {
            resultString += iter->second;
        }
        else
        {
            resultString += curChar;
        }
        i += len;
    }

    return resultString;
}

std::string bert_normalize_prompt(const std::string &text)
{
    // TODO: handle chinese characters? https://github.com/huggingface/tokenizers/blob/ef5f50605ddf9f8caef1598c0e4853862b9707a7/tokenizers/src/normalizers/bert.rs#L98
    std::string text2 = stripAccents(text);
    for (size_t i = 0; i < text2.size(); i += utf8_len(text2[i]))
    {
        char c = text2[i];
        if (c >= 'A' && c <= 'Z')
            text2[i] = c - 'A' + 'a';
    }
    return text2;
}
void bert_tokenize(
    struct bert_ctx * ctx,
    const char * text,
    bert_vocab_id * tokens,
    int32_t * n_tokens,
    int32_t n_max_tokens)
{
    int cls_tok_id = 101;
    int sep_tok_id = 102;
    const bert_vocab &vocab = ctx->vocab;

    std::string str = text;

    std::vector<std::string> words;
    // first split the text into words
    {
        str = bert_normalize_prompt(str);

        std::string pat = R"([[:punct:]]|[[:alpha:]]+|[[:digit:]]+)";

        std::regex re(pat);
        std::smatch m;

        while (std::regex_search(str, m, re))
        {
            for (std::string x : m)
            {
                words.push_back(x);
            }
            str = m.suffix();
        }
    }

    int32_t t = 0;
    tokens[t++] = cls_tok_id;

    // find the longest tokens that form the words:
    for (const auto &word : words)
    {
        if (word.size() == 0)
            continue;

        int i = 0;
        int n = word.size();
        auto *token_map = &vocab.token_to_id;
    loop:
        while (i < n)
        {
            if (t >= n_max_tokens - 1)
                break;
            int j = n;
            while (j > i)
            {
                auto it = token_map->find(word.substr(i, j - i));
                if (it != token_map->end())
                {
                    tokens[t++] = it->second;
                    i = j;
                    token_map = &vocab.subword_token_to_id;
                    goto loop;
                }
                --j;
            }
            if (j == i)
            {
                fprintf(stderr, "%s: unknown token '%s'\n", __func__, word.substr(i, 1).data());
                token_map = &vocab.subword_token_to_id;
                ++i;
            }
        }
    }
    tokens[t++] = sep_tok_id;
    *n_tokens = t;
}

//
// Loading and setup
//

struct bert_ctx * bert_load_from_file(const char *fname)
{
    printf("%s: loading model from '%s' - please wait ...\n", __func__, fname);

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin)
    {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname);
        return nullptr;
    }

    // verify magic
    {
        uint32_t magic;
        fin.read((char *)&magic, sizeof(magic));
        if (magic != 0x67676d6c)
        {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname);
            return nullptr;
        }
    }

    bert_ctx * new_bert = new bert_ctx;
    bert_model & model = new_bert->model;
    bert_vocab & vocab = new_bert->vocab;

    // load hparams
    {
        auto &hparams = model.hparams;

        fin.read((char *)&hparams.n_vocab, sizeof(hparams.n_vocab));
        fin.read((char *)&hparams.n_max_tokens, sizeof(hparams.n_max_tokens));
        fin.read((char *)&hparams.n_embd, sizeof(hparams.n_embd));
        fin.read((char *)&hparams.n_intermediate, sizeof(hparams.n_intermediate));
        fin.read((char *)&hparams.n_head, sizeof(hparams.n_head));
        fin.read((char *)&hparams.n_layer, sizeof(hparams.n_layer));
        fin.read((char *)&hparams.f16, sizeof(hparams.f16));

        printf("%s: n_vocab = %d\n", __func__, hparams.n_vocab);
        printf("%s: n_max_tokens   = %d\n", __func__, hparams.n_max_tokens);
        printf("%s: n_embd  = %d\n", __func__, hparams.n_embd);
        printf("%s: n_intermediate  = %d\n", __func__, hparams.n_intermediate);
        printf("%s: n_head  = %d\n", __func__, hparams.n_head);
        printf("%s: n_layer = %d\n", __func__, hparams.n_layer);
        printf("%s: f16     = %d\n", __func__, hparams.f16);
    }

    // load vocab
    {
        int32_t n_vocab = model.hparams.n_vocab;

        std::string word;
        // takes forever because reads bad data
        // i also skipped the last one. the real problem
        // was that i used the wrong tokenizer
        for (int i = 0; i < n_vocab; i++)
        {
            uint32_t len;
            fin.read((char *)&len, sizeof(len));

            word.resize(len);
            fin.read((char *)word.data(), len);

            if (word[0] == '#' && word[1] == '#')
            {
                vocab.subword_token_to_id[word.substr(2)] = i;
                vocab._id_to_subword_token[i] = word;
            }

            if (vocab.token_to_id.count(word) == 0)
            {
                vocab.token_to_id[word] = i;
                vocab._id_to_token[i] = word;
            }
        }
    }

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    ggml_type wtype = GGML_TYPE_COUNT;
    switch (model.hparams.f16)
    {
    case 0:
        wtype = GGML_TYPE_F32;
        break;
    case 1:
        wtype = GGML_TYPE_F16;
        break;
    case 2:
        wtype = GGML_TYPE_Q4_0;
        break;
    case 3:
        wtype = GGML_TYPE_Q4_1;
        break;
    default:
    {
        fprintf(stderr, "%s: invalid model file '%s' (bad f16 value %d)\n",
                __func__, fname, model.hparams.f16);
        bert_free(new_bert);
        return nullptr;
    }
    }

    auto &ctx = model.ctx;

    size_t model_mem_req = 0;

    {
        const auto &hparams = model.hparams;

        const int n_embd = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_max_tokens = hparams.n_max_tokens;
        const int n_intermediate = hparams.n_intermediate;
        const int n_vocab = hparams.n_vocab;

        // print all above values for debugging
        printf("%s: n_embd = %d\n", __func__, n_embd);
        printf("%s: n_layer = %d\n", __func__, n_layer);
        printf("%s: n_max_tokens = %d\n", __func__, n_max_tokens);
        printf("%s: n_intermediate = %d\n", __func__, n_intermediate);
        printf("%s: n_vocab = %d\n", __func__, n_vocab);

        // Calculate size requirements

        model_mem_req += n_embd * n_vocab * ggml_type_sizef(wtype); // word_embeddings
        model_mem_req += n_embd * 2 * ggml_type_sizef(wtype); // token_type_embeddings
        model_mem_req += n_embd * n_max_tokens * ggml_type_sizef(wtype); // position_embeddings

        model_mem_req += 2 * n_embd * ggml_type_sizef(GGML_TYPE_F32); // ln_e_*

        model_mem_req += 4 * n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // ln_*

        model_mem_req += 4 * n_layer * (n_embd * n_embd * ggml_type_sizef(wtype)); // kqvo weights
        model_mem_req += 4 * n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // kqvo bias

        model_mem_req += 2 * n_layer * (n_embd * n_intermediate * ggml_type_sizef(wtype)); // ff_*_w
        model_mem_req += n_layer * (n_intermediate * ggml_type_sizef(GGML_TYPE_F32)); // ff_i_b
        model_mem_req += n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // ff_o_b

        model_mem_req += (5 + 16 * n_layer) * 512; // object overhead
        // pooler
        model_mem_req += n_embd * n_embd * ggml_type_sizef(GGML_TYPE_F32);
        model_mem_req += n_embd * ggml_type_sizef(GGML_TYPE_F32);
        // classifier (this is why i was out of memory)
        model_mem_req += (6*n_embd) * ggml_type_sizef(GGML_TYPE_F32);
        model_mem_req += 6 * ggml_type_sizef(GGML_TYPE_F32);

        printf("%s: ggml ctx size = %6.2f MB (%d B)\n", __func__, model_mem_req / (1024.0 * 1024.0), model_mem_req);
    }

    // create the ggml context
    {
        struct ggml_init_params params = {
            .mem_size = model_mem_req,
            .mem_buffer = NULL,
            .no_alloc = false,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx)
        {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            bert_free(new_bert);
            return nullptr;
        }
    }

    printf("prepare memory\n");

    // prepare memory for the weights
    {
        const auto &hparams = model.hparams;

        const int n_embd = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_intermediate = hparams.n_intermediate;
        const int n_max_tokens = hparams.n_max_tokens;
        const int n_vocab = hparams.n_vocab;

        model.layers.resize(n_layer);

        model.word_embeddings = ggml_new_tensor_2d(ctx, wtype, n_embd, n_vocab);
        model.token_type_embeddings = ggml_new_tensor_2d(ctx, wtype, n_embd, 2);
        model.position_embeddings = ggml_new_tensor_2d(ctx, wtype, n_embd, n_max_tokens);

        model.ln_e_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        model.ln_e_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

        // map by name
        model.tensors["embeddings.word_embeddings.weight"] = model.word_embeddings;
        model.tensors["embeddings.token_type_embeddings.weight"] = model.token_type_embeddings;
        model.tensors["embeddings.position_embeddings.weight"] = model.position_embeddings;

        model.tensors["embeddings.LayerNorm.weight"] = model.ln_e_w;
        model.tensors["embeddings.LayerNorm.bias"] = model.ln_e_b;

        for (int i = 0; i < n_layer; ++i)
        {
            auto &layer = model.layers[i];

            layer.ln_att_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.ln_att_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.ln_out_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.ln_out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            layer.q_w = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.q_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.k_w = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.k_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.v_w = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.v_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.o_w = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.o_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            layer.ff_i_w = ggml_new_tensor_2d(ctx, wtype, n_embd, n_intermediate);
            layer.ff_i_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_intermediate);

            layer.ff_o_w = ggml_new_tensor_2d(ctx, wtype, n_intermediate, n_embd);
            layer.ff_o_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            // map by name

            model.tensors["encoder.layer." + std::to_string(i) + ".attention.self.query.weight"] = layer.q_w;
            model.tensors["encoder.layer." + std::to_string(i) + ".attention.self.query.bias"] = layer.q_b;
            model.tensors["encoder.layer." + std::to_string(i) + ".attention.self.key.weight"] = layer.k_w;
            model.tensors["encoder.layer." + std::to_string(i) + ".attention.self.key.bias"] = layer.k_b;
            model.tensors["encoder.layer." + std::to_string(i) + ".attention.self.value.weight"] = layer.v_w;
            model.tensors["encoder.layer." + std::to_string(i) + ".attention.self.value.bias"] = layer.v_b;
            model.tensors["encoder.layer." + std::to_string(i) + ".attention.output.LayerNorm.weight"] = layer.ln_att_w;
            model.tensors["encoder.layer." + std::to_string(i) + ".attention.output.LayerNorm.bias"] = layer.ln_att_b;
            model.tensors["encoder.layer." + std::to_string(i) + ".attention.output.dense.weight"] = layer.o_w;
            model.tensors["encoder.layer." + std::to_string(i) + ".attention.output.dense.bias"] = layer.o_b;

            model.tensors["encoder.layer." + std::to_string(i) + ".intermediate.dense.weight"] = layer.ff_i_w;
            model.tensors["encoder.layer." + std::to_string(i) + ".intermediate.dense.bias"] = layer.ff_i_b;

            model.tensors["encoder.layer." + std::to_string(i) + ".output.LayerNorm.weight"] = layer.ln_out_w;
            model.tensors["encoder.layer." + std::to_string(i) + ".output.LayerNorm.bias"] = layer.ln_out_b;
            model.tensors["encoder.layer." + std::to_string(i) + ".output.dense.weight"] = layer.ff_o_w;
            model.tensors["encoder.layer." + std::to_string(i) + ".output.dense.bias"] = layer.ff_o_b;
        }

        // classifier
        model.cls_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 6);
        model.cls_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 6);
        // map by name
        model.tensors["classifier.weight"] = model.cls_w;
        model.tensors["classifier.bias"] = model.cls_b;

        // pooler
        model.plr_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);
        model.plr_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        // map by name
        model.tensors["pooler.dense.weight"] = model.plr_w;
        model.tensors["pooler.dense.bias"] = model.plr_b;
    }

 printf("load weights\n");
    // load weights
    {
        int n_tensors = 0;
        size_t total_size = 0;

        printf("%s: ", __func__);

        while (true)
        {
            int32_t n_dims;
            int32_t length;
            int32_t ftype;

            fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            fin.read(reinterpret_cast<char *>(&length), sizeof(length));
            fin.read(reinterpret_cast<char *>(&ftype), sizeof(ftype));

            if (fin.eof())
            {
                break;
            }

            int64_t nelements = 1;
            int64_t ne[2] = {1, 1};
            for (int i = 0; i < n_dims; ++i)
            {
                int32_t ne_cur;
                fin.read(reinterpret_cast<char *>(&ne_cur), sizeof(ne_cur));
                ne[i] = ne_cur;
                nelements *= ne[i];
            }

            std::string name(length, 0);
            fin.read(&name[0], length);

            if (model.tensors.find(name.data()) == model.tensors.end())
            {
                fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
                bert_free(new_bert);
                return nullptr;
            }

            auto tensor = model.tensors[name.data()];
            if (ggml_nelements(tensor) != nelements)
            {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                bert_free(new_bert);
                return nullptr;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1])
            {
                fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%lld, %lld], expected [%lld, %lld]\n",
                        __func__, name.data(), tensor->ne[0], tensor->ne[1], ne[0], ne[1]);
                bert_free(new_bert);
                return nullptr;
            }

            if (0)
            {
                static const char *ftype_str[] = {
                    "f32",
                    "f16",
                    "q4_0",
                    "q4_1",
                };
                printf("%24s - [%5lld, %5lld], type = %6s, %6.2f MB, %9zu bytes\n", name.data(), ne[0], ne[1], ftype_str[ftype], ggml_nbytes(tensor) / 1024.0 / 1024.0, ggml_nbytes(tensor));
            }

            size_t bpe = 0;

            switch (ftype)
            {
            case 0:
                bpe = ggml_type_size(GGML_TYPE_F32);
                break;
            case 1:
                bpe = ggml_type_size(GGML_TYPE_F16);
                break;
            case 2:
                bpe = ggml_type_size(GGML_TYPE_Q4_0);
                assert(ne[0] % 64 == 0);
                break;
            case 3:
                bpe = ggml_type_size(GGML_TYPE_Q4_1);
                assert(ne[0] % 64 == 0);
                break;
            default:
            {
                fprintf(stderr, "%s: unknown ftype %d in model file\n", __func__, ftype);
                bert_free(new_bert);
                return nullptr;
            }
            };

            if ((nelements * bpe) / ggml_blck_size(tensor->type) != ggml_nbytes(tensor))
            {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %llu\n",
                        __func__, name.data(), ggml_nbytes(tensor), nelements * bpe);
                bert_free(new_bert);
                return nullptr;
            }

            fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));

            // printf("%42s - [%5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], ftype == 0 ? "float" : "f16", ggml_nbytes(tensor)/1024.0/1024.0);
            total_size += ggml_nbytes(tensor);
            if (++n_tensors % 8 == 0)
            {
                printf(".");
                fflush(stdout);
            }
        }

        printf(" done\n");

        printf("%s: model size = %8.2f MB / num tensors = %d\n", __func__, total_size / 1024.0 / 1024.0, n_tensors);
    }

    fin.close();

    // Calculate space requirements for setting up context buffers later
    {
        bert_vocab_id tokens[] = {0, 1, 2, 3};
        // TODO: We set the initial buffer size to 32MB and hope it's enough. Maybe there is a better way to do this?
        new_bert->buf_compute.resize(32 * 1024 * 1024);
        // bert_eval(new_bert, 1, tokens, 4, nullptr);
        printf("finished\n");
        new_bert->max_batch_n = 0;

        // TODO: Max tokens should be a param?
        int32_t N = 512;//new_bert->model.hparams.n_max_tokens;
        new_bert->mem_per_input = 1.1 * (new_bert->mem_per_token * N); // add 10% to account for ggml object overhead

    }
    printf("%s: mem_per_token %zu KB, mem_per_input %lld MB\n", __func__, new_bert->mem_per_token / (1 << 10), new_bert->mem_per_input / (1 << 20));

    return new_bert;
}

void bert_resize_ctx(bert_ctx * ctx, int32_t new_size) {    
    int64_t buf_size_new = ctx->mem_per_input * new_size;

    // TODO: Max memory should be a param? Now just 1 GB
    int64_t GB = 1 << 30;
    //printf("%s: requested_buf_size %lldMB\n", __func__, buf_size_new / (1 << 20));
    if (buf_size_new > GB) {
        int32_t adjusted_new_size = GB / ctx->mem_per_input;
        if (adjusted_new_size < 1) adjusted_new_size = 1;
        //printf("%s: requested batch size %d, actual new batch size %d\n", __func__, new_size, adjusted_new_size);
        new_size = adjusted_new_size;
        buf_size_new = ctx->mem_per_input * new_size;
    }
    if (new_size > ctx->max_batch_n) {
        ctx->buf_compute.resize(buf_size_new);
        ctx->max_batch_n = new_size;
    }
}

void bert_free(bert_ctx * ctx) {
    ggml_free(ctx->model.ctx);
    delete ctx;
}

void bert_eval(
    struct bert_ctx *ctx,
    int32_t n_threads,
    bert_vocab_id *tokens,
    int32_t n_tokens,
    float *embeddings)
{
    bert_eval_batch(ctx, n_threads, 1, &tokens, &n_tokens, embeddings ? &embeddings : nullptr);
}

void bert_eval_batch(
    bert_ctx * ctx,
    int32_t n_threads,
    int32_t n_batch_size,
    bert_vocab_id ** batch_tokens,
    int32_t * n_tokens,
    float ** batch_embeddings)
{
    const bert_model& model = ctx->model;
    bool mem_req_mode = !batch_embeddings;
    // batch_embeddings is nullptr for the initial memory requirements run
    if (!mem_req_mode && n_batch_size > ctx->max_batch_n) {
        bert_resize_ctx(ctx, n_batch_size);
        if (n_batch_size > ctx->max_batch_n) {
            fprintf(stderr, "%s: tried to increase buffers to batch size %d but failed\n", __func__, n_batch_size);
            return;
        }
    }

    printf("tokens[0]: %d; batch: %d\n", n_tokens[0], n_batch_size);

    // TODO: implement real batching
    int ba = 0;
    // for (int ba = 0; ba < n_batch_size; ba++)
    // {
        printf("------------ batch -----------\n");
        const int N = n_tokens[ba];
        const auto &tokens = batch_tokens[ba];
        printf("tokens: (%d) ", N);
        for (int j = 0; j < N; j++) {
            printf("%d ", tokens[j]);
        }
        printf("\n");

        const auto &hparams = model.hparams;

        const int n_embd = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_max_tokens = hparams.n_max_tokens;
        const int n_head = hparams.n_head;

        const int d_head = n_embd / n_head;

        std::vector<float> result;
        if (N > n_max_tokens)
        {
            fprintf(stderr, "Too many tokens, maximum is %d\n", n_max_tokens);
            return;
        }

        auto & mem_per_token = ctx->mem_per_token;
        auto & buf_compute   = ctx->buf_compute;

        struct ggml_init_params params = {
            .mem_size = buf_compute.size,
            .mem_buffer = buf_compute.data,
            .no_alloc = false,
        };

        struct ggml_context *ctx0 = ggml_init(params);
        struct ggml_cgraph gf = {};

        // Embeddings. word_embeddings + token_type_embeddings + position_embeddings
        struct ggml_tensor *token_layer = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
        memcpy(token_layer->data, tokens, N * ggml_element_size(token_layer));

        struct ggml_tensor *token_types = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
        ggml_set_zero(token_types);

        struct ggml_tensor *positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
        for (int i = 0; i < N; i++)
        {
            ggml_set_i32_1d(positions, i, i);
        }

        struct ggml_tensor *inpL = ggml_get_rows(ctx0, model.word_embeddings, token_layer);
        ggml_tensor *t1 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 8);
        for (int i = 0; i < 8; i++)
        {
            ggml_set_i32_1d(t1, i, 1);
        }
        ggml_tensor *t2 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 8);
        for (int i = 0; i < 8; i++)
        {
            ggml_set_i32_1d(t2, i, 2);
        }
        ggml_tensor *t3 = ggml_get_rows(ctx0, t1, t2);
        printf("embedding: (%d, %d, %d, %d)\n", t3->ne[0], t3->ne[1], t3->ne[2], t3->ne[3]);
        for (size_t j = 0; j < 8; j++)
        {
            printf("%f, ", ggml_get_f32_nd(t3, j, j, 0, 0));
            // printf("%d", (((int**)(t3->data))[0])[0]);
        }
        printf("...\n");

        inpL = ggml_add(ctx0,
                        ggml_get_rows(ctx0, model.token_type_embeddings, token_types),
                        inpL);
        inpL = ggml_add(ctx0,
                        ggml_get_rows(ctx0, model.position_embeddings, positions),
                        inpL);

        printf("embedding: (%d, %d, %d, %d): [", inpL->ne[0], inpL->ne[1], inpL->ne[2], inpL->ne[3]);

        // embd norm
        {
            inpL = ggml_norm(ctx0, inpL, 1);

            inpL = ggml_add(ctx0,
                            ggml_mul(ctx0,
                                     ggml_repeat(ctx0, model.ln_e_w, inpL),
                                     inpL),
                            ggml_repeat(ctx0, model.ln_e_b, inpL));
        }

        /*
        hidden_states[0][0][:8]=tensor([ 0.0882,  0.0976, -0.1284, -0.4466,  0.2492,  0.1160, -0.3225,  0.1198])
hidden_states[0][0][:8]=tensor([-0.0238, -0.2069, -0.3990, -0.2829,  0.1547,  0.0127, -0.2391,  0.0076])
hidden_states[0][0][:8]=tensor([ 0.0419, -0.2978, -0.1759, -0.0502,  0.4118, -0.2959, -0.1616, -0.0482])
hidden_states[0][0][:8]=tensor([ 0.3975, -0.5446, -0.4481, -0.3630, -0.0162, -0.2981, -0.2700,  0.1272])
hidden_states[0][0][:8]=tensor([ 0.0294, -0.6950, -0.3598, -0.5588, -0.2643,  0.2164, -0.1816,  0.2524])
hidden_states[0][0][:8]=tensor([-0.0550, -1.1536, -0.3987, -1.1224, -0.3142,  0.3142, -0.0529,  0.1895])
hidden_states[0][0][:8]=tensor([-0.0655, -1.0673, -0.8218, -0.1538,  0.2745,  0.6265,  0.2034,  0.3209])
hidden_states[0][0][:8]=tensor([-0.0796, -0.7166, -0.7001, -0.2648,  0.0547,  0.3296,  0.0299, -0.0764])
hidden_states[0][0][:8]=tensor([-0.4792, -0.2945, -0.5080, -0.1174, -0.0301,  0.3388,  0.1514, -0.1627])
hidden_states[0][0][:8]=tensor([-0.8876, -0.4008, -0.0088, -0.0713, -0.1203,  0.0215, -0.1082,  0.5719])
hidden_states[0][0][:8]=tensor([-0.9155, -0.2493, -0.3955, -0.5377, -0.0996,  0.2699, -0.3798,  1.2915])
hidden_states[0][0][:8]=tensor([-0.9989, -1.0350, -0.7127, -0.7082,  0.4737, -0.9528,  0.5832,  0.6503])
        */

        // layers
        printf("layers: %d\n", n_layer);
        for (int il = 0; il < n_layer; il++)
        {
            struct ggml_tensor *cur = inpL;

            // self-attention
            {
                struct ggml_tensor *Qcur = cur;
                Qcur = ggml_reshape_3d(ctx0,
                                       ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].q_b, cur),
                                                ggml_mul_mat(ctx0, model.layers[il].q_w, cur)),
                                       d_head, n_head, N);
                struct ggml_tensor *Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);

                struct ggml_tensor *Kcur = cur;
                Kcur = ggml_reshape_3d(ctx0,
                                       ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].k_b, Kcur),
                                                ggml_mul_mat(ctx0, model.layers[il].k_w, Kcur)),
                                       d_head, n_head, N);
                struct ggml_tensor *K = ggml_permute(ctx0, Kcur, 0, 2, 1, 3);

                struct ggml_tensor *Vcur = cur;
                Vcur = ggml_reshape_3d(ctx0,
                                       ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].v_b, Vcur),
                                                ggml_mul_mat(ctx0, model.layers[il].v_w, Vcur)),
                                       d_head, n_head, N);
                struct ggml_tensor *V = ggml_permute(ctx0, Vcur, 0, 2, 1, 3);

                struct ggml_tensor *KQ = ggml_mul_mat(ctx0, K, Q);
                // KQ = soft_max(KQ / sqrt(head width))
                KQ = ggml_soft_max(ctx0,
                                   ggml_scale(ctx0,
                                              KQ,
                                              ggml_get_data_f32(ggml_new_f32(ctx0, 1.0f / sqrt((float)d_head)))[0]));

                /*
                 # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
                */

                // dropout ? attention probs
                // \/ V = attention probs @ V

                V = ggml_cont(ctx0, ggml_transpose(ctx0, V));
                struct ggml_tensor *KQV = ggml_mul_mat(ctx0, V, KQ); // ctx layer
                KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

        //         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        // new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        // context_layer = context_layer.view(new_context_layer_shape)

        // done in attention
        /* ???
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs
        */

                float *outs34 = (float*)ggml_get_data(KQV);
                printf("layer %d KQV: (%d, %d, %d, %d): [", il, inpL->ne[0], inpL->ne[1], inpL->ne[2], inpL->ne[3]);
                for (size_t j = 0; j < 8; j++)
                {
                    printf("%f, ", outs34[j]);
                }
                printf("...\n");

                cur = ggml_cpy(ctx0,
                               KQV,
                               ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));
            }
            // attention output
            cur = ggml_add(ctx0,
                           ggml_repeat(ctx0, model.layers[il].o_b, cur),
                           ggml_mul_mat(ctx0, model.layers[il].o_w, cur));

            // re-add the layer input
            cur = ggml_add(ctx0, cur, inpL);

            // attention norm
            {
                cur = ggml_norm(ctx0, cur, 0);

                cur = ggml_add(ctx0,
                               ggml_mul(ctx0,
                                        ggml_repeat(ctx0, model.layers[il].ln_att_w, cur),
                                        cur),
                               ggml_repeat(ctx0, model.layers[il].ln_att_b, cur));
            }
            struct ggml_tensor *att_output = cur;
            // intermediate_output = self.intermediate(attention_output)
            att_output = ggml_mul_mat(ctx0, model.layers[il].ff_i_w, att_output);
            att_output = ggml_add(ctx0,
                           ggml_repeat(ctx0, model.layers[il].ff_i_b, att_output),
                           att_output);
            att_output = ggml_gelu(ctx0, att_output);

            // layer_output = self.output(intermediate_output, attention_output)
            att_output = ggml_mul_mat(ctx0, model.layers[il].ff_o_w, att_output);
            att_output = ggml_add(ctx0,
                           ggml_repeat(ctx0, model.layers[il].ff_o_b, att_output),
                           att_output);
            // attentions bypass the intermediate layer
            cur = ggml_add(ctx0, att_output, cur);

            // output norm
            {
                cur = ggml_norm(ctx0, cur, 0);

                cur = ggml_add(ctx0,
                               ggml_mul(ctx0,
                                        ggml_repeat(ctx0, model.layers[il].ln_out_w, cur),
                                        cur),
                               ggml_repeat(ctx0, model.layers[il].ln_out_b, cur));
            }
            inpL = cur;

            float *outs33 = (float*)ggml_get_data(inpL);
            printf("layer %d (%s): (%d, %d, %d, %d): [", il, *inpL->name, inpL->ne[0], inpL->ne[1], inpL->ne[2], inpL->ne[3]);
            for (size_t j = 0; j < 8; j++)
            {
                printf("%f, ", outs33[j]);
            }
            printf("]\n");
        }

        // float *outs1 = ggml_get_data_f32(inpL);
        // ---------------------- something here is wrong ----------------------
        // print the output
        // printf("layers: (%d, %d, %d, %d): ", inpL->ne[0], inpL->ne[1], inpL->ne[2], inpL->ne[3]);
        // for (size_t j = 0; j < 8; j++)
        // {
        //     printf("%f ", outs1[j]);
        // }
        // printf("...\n");

        // print the output
        // ggml_tensor *inpl2 = ggml_transpose(ctx0, inpL);
        // printf("transpose: (%d, %d, %d, %d): ", inpl2->ne[0], inpl2->ne[1], inpl2->ne[2], inpl2->ne[3]);
        // float *outs2 = ggml_get_data_f32(inpl2);
        // for (size_t j = 0; j < 8; j++)
        // {
        //     printf("%f ", outs2[j]);
        // }
        // printf("...\n");

        inpL = ggml_cont(ctx0, ggml_transpose(ctx0, inpL));
        printf("cont: (%d, %d, %d, %d): ", inpL->ne[0], inpL->ne[1], inpL->ne[2], inpL->ne[3]);
        float *outs8 = ggml_get_data_f32(inpL);
        for (size_t j = 0; j < inpL->ne[0]; j++)
        {
            printf("%f ", outs8[j]);
        }
        printf("...\n");

        // pooler
        struct ggml_tensor *sum2 = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, inpL->ne[0], inpL->ne[0]);
        ggml_tensor *inpL2 = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, inpL->ne[0], inpL->ne[1]);
        ggml_set_f32(inpL2, 1.0f);
        // ggml_tensor *inpL2 = inpL;//ggml_mul_mat(ctx0, sum2, inpL);

        struct ggml_tensor *sum = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, inpL->ne[0], inpL->ne[1]);
        ggml_set_f32(sum, 1.0f / N);
        struct ggml_tensor *inpL3 = ggml_mul_mat(ctx0, sum, inpL2);
        

        float *outs344 = ggml_get_data_f32(inpL3);
        printf("\n%d\n", inpL->backend);
        printf("cpy: (%d, %d, %d, %d): [", inpL3->ne[0], inpL3->ne[1], inpL3->ne[2], inpL3->ne[3]);
        for (size_t j = 0; j < 8; j++)
        {
            printf("%f, ", outs344[j]);
        }
        printf("]\n");
        // TODO maybe dropout effects other dimenstion
        // TODO cant multiply, need to check
        // TODO memory issue, must be due to wrong sizes
        //     was using clr as plr, the memory is not enough
        // this has a memroy issue
        struct ggml_tensor *test = ggml_transpose(ctx0, inpL);
        float *outs35 = ggml_get_data_f32(test);
        // // print the output
        printf("pooler: (%d, %d, %d, %d): [", test->ne[0], test->ne[1], test->ne[2], test->ne[3]);
        for (size_t j = 0; j < 8; j++)
        {
            printf("%f, ", outs35[j]);
        }
        printf("]\n");
        // test = ggml_mul_mat(ctx0, model.plr_w, ggml_transpose(ctx0, inpL)); // not sure about this
        test = ggml_add(ctx0, ggml_mul_mat(ctx0, model.plr_w, ggml_transpose(ctx0, inpL)), model.plr_b);
        test = ggml_tanh(ctx0, test);

        float *outs3 = ggml_get_data_f32(test);
        // // print the output
        printf("pooler: (%d, %d, %d, %d): [", inpL->ne[0], inpL->ne[1], inpL->ne[2], inpL->ne[3]);
        for (size_t j = 0; j < 8; j++)
        {
            printf("%f, ", outs3[j]);
        }
        printf("]\n");

        // dropout
        // float dropout = 0.1f; // <- this is what bert classifier uses
        // // create a tensor of random values that is the size of n_embd
        // struct ggml_tensor *rand_tensor = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, inpL->ne[0]);
        // // fill with random numbers between 0 and 1
        // for (int i = 0; i < inpL->ne[0]; i++)
        // {
        //     // not sure what this is doing exatly (and not as cool as grabbing memory), but ok
        //     ggml_set_f32_1d(rand_tensor, i, (float)rand() / RAND_MAX);
        // }

        // set all above the dropout to 1 and those below to 0
        // float *rand_floats = ggml_get_data_f32(rand_tensor);
        // int dropped = 0;
        // for (int j = 0; j < inpL->ne[0]; j++)
        // {
        //     // does >= matter
        //     ggml_set_f32_1d(rand_tensor, j, rand_floats[j] > dropout ? 1.0 : 0.0);
        //     dropped += rand_floats[j] > dropout ? 0 : 1;
        // }

        // printf("rand (%d); dropped = %d : ", inpL->ne[0], dropped);
        // for (size_t j = 0; j < 8; j++)
        // {
        //     printf("%f ", rand_floats[j]);
        // }
        // printf("...\n");

        // // multiply inpL by rand_tensor
        // inpL = ggml_mul(ctx0, inpL, rand_tensor);
        // // divide inpL by dropout
        // // inpL = ggml_scale(ctx0, inpL, 1.0f / (1.0f - dropout));
        // // inpL = ggml_scale(ctx0, inpL, dropout);
        // float *inpL_data = ggml_get_data_f32(inpL);
        // for (size_t j = 0; j < inpL->ne[0]; j++)
        // {
        //     ggml_set_f32_1d(inpL, j, inpL_data[j] / dropout);
        // }
        // // print the output
        // printf("drop: [ ");
        // float *my_outs1 = ggml_get_data_f32(inpL);
        //     for (size_t j = 0; j < 8; j++)
        //     {
        //         printf("%f, ", my_outs1[j]);
        //     }
        //     printf("]\n");

        // float my_pys[] = {-0.2972660959, -0.5068098903, -0.9352156520,  0.1820614785,
        //  0.7976894975, -0.4918534160, -0.3715885580,  0.4166602492,
        // -0.9677098393, -0.9983722568, -0.8730065823,  0.8252849579,
        //  0.8025905490,  0.6008809805,  0.0101931887, -0.3116634190,
        // -0.3111754954, -0.4933052957,  0.7215374112,  0.8287019134,
        //  0.0329882614,  0.9999914765, -0.6611698866,  0.4561907649,
        //  0.7409261465,  0.9362468123, -0.4562022686, -0.0362270772,
        //  0.4030611813,  0.3491962850,  0.1712955534,  0.5320036411,
        // -0.9242640734, -0.4158183038, -0.9744039774, -0.9218249321,
        //  0.6224009395, -0.0817109793, -0.3311074078, -0.4076958895,
        // -0.6468805671,  0.7521478534,  0.9998099208,  0.6495472789,
        //  0.7948874235, -0.5147274137, -0.9999787807,  0.6779716015,
        //  0.0851315781,  0.9749540091,  0.8841860294,  0.9617276192,
        //  0.5086355209,  0.5947693586,  0.6801412106, -0.4805503190,
        //  0.2430445105,  0.5363256335, -0.5015187263, -0.6854358912,
        // -0.6462809443,  0.5367029905, -0.7826506495, -0.6138799191,
        //  0.9758753181,  0.9183585644, -0.6651331782, -0.6195915341,
        // -0.1615225226,  0.3777918220,  0.0432123505,  0.3741019368,
        // -0.0545742698, -0.1902745217,  0.8450708389,  0.6824842095,
        // -0.6934248209,  1.0000000000, -0.3845966458, -0.7702664733,
        //  0.8181520104,  0.8211403489,  0.6464481354, -0.8203750849,
        //  0.4614720643, -0.9999997616,  0.5557513833, -0.4914001524,
        // -0.8035466671,  0.6214201450,  0.6133602262, -0.4610658586,
        //  0.6249681711,  0.6430872083, -0.4636088014, -0.5815607905,
        // -0.5374680161, -0.9537124038, -0.7585250735, -0.8491303325,
        //  0.6254301071, -0.6298698187, -0.6381314397, -0.5392715931,
        //  0.7719419003, -0.7058637142,  0.4938770831,  0.4252541065,
        //  0.2453287840,  0.6281377077,  0.4380756021, -0.7117482424,
        //  0.7855562568, -0.4487775862,  0.5972908735, -0.6520398855,
        // -0.8463907838, -0.6099019647, -0.8263155818,  0.2218839526,
        // -0.5077425838, -0.7127110958, -0.3604827523, -0.5998253822,
        //  0.5561587214, -0.4129746854, -0.9730507135, -1.0000000000,
        // -0.6645724773, -0.5866327882,  0.0325824134, -0.5091377497,
        // -0.7426586151, -0.6644917727,  0.8214862943,  0.5218549371,
        //  0.4349996746,  0.9993069768, -0.3473171294,  0.4245658517,
        // -0.5365082026, -0.7449006438,  0.8607290983, -0.7082146406,
        //  0.7779422998, -0.4894568324, -0.2254562974,  0.3566251695,
        // -0.4408530295,  0.6410461664, -0.5381599665, -0.4433560371,
        // -0.9165158272, -0.2656996548, -0.5853997469,  0.0168032702,
        // -0.7713560462, -0.9625664353, -0.5052155852, -0.3336802721,
        // -0.4332163632,  0.4344982803,  0.6842092872,  0.5032737255,
        // -0.4996045828,  0.7549920082, -0.1054353490,  0.5571967959,
        // -0.1812892556, -0.2768427134,  0.5396714807, -0.5913781524,
        // -0.9079537988, -0.7932139039, -0.5196439624,  0.1490546316,
        //  0.8700690866,  0.3686432838,  0.8137699366,  0.8342354894,
        // -0.5102357864,  0.7462788224, -0.8300424218,  0.8691446781,
        // -0.5473065972,  0.5938234925, -0.6814464331,  0.7107712030,
        // -0.0943354294,  0.0931738988,  0.1170233414, -0.7710343599,
        // -0.3844282627, -0.3245749772, -0.7203885317, -0.5381140113,
        // -0.8466525674,  0.2192056179, -0.6466286778, -0.7276512384,
        // -0.5858774185,  0.4399649203,  0.2574832737, -0.0633125454,
        //  0.7886621952,  0.2286065221, -0.0628866106, -0.0504529141,
        //  0.4773051143,  0.5439264774,  0.7116459608,  0.8693832159,
        // -0.5887288451, -0.4497594833, -0.3490993679, -0.7313286662,
        //  0.4441732168, -0.4893394709, -0.5397129655, -0.7476241589,
        //  0.7691163421, -0.3978340924,  0.4684024751,  0.3996479213,
        //  0.6340875030, -0.1896305978,  0.4988335967, -0.7290810347,
        //  0.4850661159, -0.5945520401,  0.9708783031,  0.9161819220,
        // -0.6428366899, -0.8275576830,  0.8947751522, -0.8042297363,
        // -0.6555616856, -0.3356007636, -0.5218035579,  0.1015272513,
        // -0.6154671907,  0.6709885597,  0.8920181394,  0.3900047541,
        //  0.0838884786, -0.6811017394, -0.1124921888, -0.5840511322,
        // -0.6080525517,  0.1772038788,  0.9530708194,  0.5617136359,
        //  0.5959689617, -0.0974573866, -0.5159488916, -0.2374396622,
        // -0.9739671350, -0.6912425756, -0.9637411833, -0.5560829043,
        // -0.8860992789,  0.8644613624,  0.5469789505,  0.5993382931,
        // -0.7693706751, -0.3494363129, -0.5768139362,  0.3428893387,
        //  0.4295089245, -0.3233615458, -0.3240713775, -0.3086241484,
        // -0.6256707907, -0.7216758728,  0.4381043017, -0.3711259663,
        // -0.7547987103,  0.4191041589, -0.2834304273,  0.5200014710,
        //  0.2245791107,  0.5231721997, -0.9167224765,  0.7866622210,
        //  0.9999998808,  0.8336402774,  0.0124949310, -0.2737207413,
        // -0.9999116659, -0.9503225684,  0.9993046522, -0.9759051800,
        // -0.9999999404, -0.2720004320, -0.5967354178, -0.0522511303,
        // -1.0000000000, -0.5774415731, -0.2061988562, -0.4330701232,
        //  0.8364667296,  0.6804265380, -0.3076015711, -1.0000000000,
        //  0.1430687308,  0.1454736143, -0.7016770840,  0.8883524537,
        // -0.6685402393,  0.6028012037,  0.4568028152,  0.3343020678,
        // -0.4380013943,  0.7077580094, -0.9330867529, -0.4647400677,
        // -0.8157466650, -0.8159846663,  0.9985263348,  0.4740940928,
        // -0.4823633432,  0.0176648404,  0.5538682938, -0.3664211035,
        //  0.3125224411, -0.5962881446, -0.6499747634,  0.6096019149,
        //  0.5934240818,  0.3792133927,  0.5027255416,  0.0391947702,
        //  0.6926938295,  0.4602409005, -0.5634712577,  0.5028960705,
        // -0.5450002551,  0.4610600471, -0.7124657631,  0.6376274824,
        // -0.7675316930, -0.7895125747,  0.5916177034, -0.4881218076,
        //  0.9631227255,  1.0000000000,  0.8645472527,  0.1604047865,
        //  0.0842669308,  0.6464627385, -0.2407859415,  0.9999999404,
        //  0.8276001811, -0.7883334160, -0.5991669297,  0.6914964914,
        // -0.7718745470, -0.7862144113,  0.9882781506, -0.5172873735,
        // -0.9411110878, -0.5709692836,  0.9428257942, -0.8833041191,
        //  0.9964423180,  0.2916823626, -0.6433544755,  0.3380635083,
        //  0.3955040574, -0.5025187135, -0.6863433123,  0.2263695747,
        // -0.6002386212,  0.3941930532, -0.0263953283,  0.2498129606,
        // -0.0352972262, -0.4731802046,  0.3778951764,  0.1389530450,
        // -0.6552792192,  0.4783546627, -0.8358768225, -0.5471740365,
        //  0.8935155869,  0.5014074445, -0.5118756890,  0.0505204946,
        // -0.6583809257, -0.9004225731, -0.3002643585,  0.6723323464,
        //  1.0000000000, -0.2690084279,  0.9314432740, -0.2616189122,
        // -0.3101498485,  0.4430228174,  0.6450682282,  0.6941943169,
        // -0.6782219410, -0.5395509005,  0.9442227483,  0.1476597339,
        // -0.9155358076, -0.2429571748,  0.6332929134, -0.4576994777,
        //  0.9994155765,  0.4928852320,  0.7261959910,  0.7647956014,
        //  0.9730598330,  0.4969083071, -0.3059513271,  0.9484713078,
        //  0.8852885962, -0.6060662866,  0.5796439052, -0.4009802938,
        // -0.9663059711, -0.5059092045, -0.5202379823,  0.4915649891,
        // -0.8935397863, -0.3653744757, -0.5101255774,  0.7635875940,
        //  0.9897100925,  0.5838787556,  0.6264415383,  0.8730959296,
        //  1.0000000000, -0.9825278521,  0.2764940858,  0.7845236063,
        // -0.4080223143, -0.9998971820,  0.6519755721, -0.5759445429,
        // -0.5481263995, -0.8837832808, -0.5477829576,  0.4381004870,
        // -0.7008681297,  0.9060137868,  0.3494331837, -0.5400179029,
        // -0.7124781609, -0.8939713240, -0.3686172962,  0.4538900256,
        // -0.9891640544, -0.4234201610, -0.2234286964,  0.5857742429,
        // -0.6291439533, -0.3420516253, -0.7706760168, -0.6431802511,
        //  0.5482181311, -0.5807697177,  0.7277950048,  0.9581997991,
        //  0.4352462888, -0.9457518458, -0.0113242669, -0.2796343565,
        // -0.5575990677,  0.3046801984, -0.4250124991, -0.9075073004,
        // -0.5942471623,  1.0000000000, -0.4583260417,  0.8422636390,
        // -0.0662450120,  0.2110111862, -0.4958843887,  0.3245919645,
        //  0.9314925671,  0.6634361744, -0.5256189108, -0.9660092592,
        //  0.7605926394, -0.5342369080,  0.5281631351,  0.8132052422,
        //  0.5780954361,  0.5827391744,  0.9412736893,  0.5875340700,
        // -0.3881354034,  0.1781292856,  0.6642351151, -0.6091355085,
        // -0.3567331135, -0.1727155894, -0.5002273917, -0.5861830115,
        //  0.6664991379,  0.9999999404,  0.5309585929,  0.5503674746,
        // -0.8428874612, -0.8505411744,  0.1425135285,  0.9999997020,
        //  0.5039612651,  0.3715060353,  0.2653190196,  0.5979276299,
        // -0.5836413503,  0.1840392649, -0.6813117862, -0.5556727052,
        //  0.5292692184,  0.3754338026,  0.5222234726, -0.3586319089,
        // -0.8629502058, -0.6740380526,  0.6584561467, -0.5608906150,
        //  0.9999304414, -0.7511833906, -0.4542506933, -0.7043898702,
        // -0.2331919968, -0.9653915167,  0.1375067085, -0.6219647527,
        // -0.3834695220,  0.7018867731,  0.4142854214,  0.6902582049,
        // -0.4978451729, -0.0799146369,  0.9613815546,  0.8196563721,
        // -0.9530281425, -0.5353205204,  0.4698372185, -0.4685251117,
        //  0.7231239080,  0.9999988675,  0.7653667331,  0.6787484288,
        //  0.3903948367, -0.0809339061,  0.4653822780, -0.8106201887,
        // -0.0592379458, -0.1290746480, -0.5447919369, -0.5485195518,
        //  0.6425699592, -0.5013921857, -0.8758537173, -0.0956273749,
        //  0.5300032496, -0.6016504169, -0.6183548570, -0.2893929780,
        //  0.2752390504, -0.0086841611, -0.5782732368, -0.5797036290,
        //  0.6379421353, -0.4373971522,  0.4916216135, -0.7650477886,
        // -0.5999747515, -0.9999956489,  0.2738643587, -1.0000000000,
        //  0.6764776707,  0.3836533725, -0.5046793818,  0.5934074521,
        //  0.6579575539,  0.4230257571, -0.0494811088, -0.9431256652,
        // -0.1945034564,  0.0643996820, -0.6616678238, -0.6826295853,
        //  0.0399110317,  0.3621577024, -0.4545105994,  0.4916269183,
        // -0.8569460511,  0.7952873111, -0.5642367005,  1.0000000000,
        //  0.3552717566, -0.6053394675,  0.3016674519,  0.4718235433,
        // -0.5415710211,  0.9999998808,  0.4242476523, -0.7027274966,
        //  0.4444245100, -0.5908246636, -0.2947342992,  0.6900447011,
        //  0.6151877046, -0.5938940644, -0.9255942702, -0.5948035121,
        // -0.0648598671, -0.4383058846,  0.6308501959, -0.5159037709,
        // -0.3390837312,  0.6659817696,  0.9445692301,  0.7791208029,
        //  0.3782727420,  0.1541347802, -0.7061821222, -0.4677431583,
        //  0.4773820937,  0.6940871477, -0.4368639886,  0.5317245126,
        //  0.9999998808,  0.5244795084, -0.4544449151, -0.5072234869,
        //  0.0405012891, -0.6486541033, -0.5493789315,  0.4954075515,
        //  0.6905415654,  0.7070558667, -0.6826462150,  0.4312375784,
        // -0.8661466837,  0.3789428771, -0.7306370735, -0.8967990279,
        //  0.6271743774,  0.1605325937, -0.7962152958, -0.7057901025,
        //  0.7167892456, -0.6233543158, -0.4362397790,  0.4671843946,
        //  0.5479197502,  0.6548162699,  0.6647970080, -0.9999999404,
        //  0.7257396579,  0.6577867270,  0.9040418267,  0.5392763615,
        //  0.7040100694,  0.7446669340,  0.6330942512, -0.7482091188,
        //  0.5066342354, -0.5964966416, -0.5239135027,  0.0443604551,
        //  0.8284822702,  0.3757095635,  0.5414506793, -0.7478888035,
        // -0.0977654904, -0.5975896120, -0.9411759377, -0.9080741405,
        //  0.5943909883, -0.8955866098,  0.2351220101,  0.6598119736,
        //  0.2909106612, -0.4764119387, -0.6468009949, -0.8729975224,
        // -0.6662276983,  0.0201338250,  0.1569862962,  0.3694507480,
        //  0.2498389035,  0.1648712307,  0.0749768987,  0.7680927515,
        // -0.9211000204,  0.0236151181, -0.8913904428,  0.5564534664,
        //  0.9727322459, -0.8784356713,  0.5811949968,  0.5723868012,
        // -0.5671886206,  0.5886072516, -0.6842981577,  0.3822332919,
        //  0.9623975754, -0.5809562206,  0.3579620421, -0.4211101830,
        // -0.3259503543, -0.6446705461, -0.5280454159, -0.6776939631,
        // -0.4136534035,  0.5546535254,  0.0100834342,  0.4498509467,
        //  0.8378413320, -0.6269623637, -0.4311333597, -0.6550615430,
        // -0.6808034778, -0.7628643513,  0.0149800396, -0.3212104440,
        // -0.3170642555,  0.7713792920,  0.1228505373,  0.9541559219,
        //  0.2775575221, -0.5483245850, -0.3800149560, -0.6853823662,
        // -0.1871784776, -0.6249330044, -0.8091664314, -0.6149207950,
        //  0.5521540642,  0.6308993697,  0.9999963641, -0.8696618080,
        // -0.9287207723, -0.5480740666, -0.4548923969,  0.6410998702,
        // -0.4031042159, -1.0000000000,  0.5889862776, -0.7710824609,
        //  0.6961896420, -0.8378028274,  0.7299239039, -0.7408285141,
        // -0.3654705286, -0.7300665975,  0.8713445663,  0.8278868198,
        // -0.5555101633, -0.7803456187,  0.3433900476, -0.8611911535,
        //  0.9402037859,  0.0175931696, -0.2899145484, -0.0466015227,
        //  0.6052978635, -0.8970783353, -0.7371670604,  0.3366930485};

        /*
        
        layer 0 ((null)): (768, 28, 1, 1): [0.088279, 0.098321, -0.128040, -0.445658, 0.248805, 0.115830, -0.322028, 0.119270, ]
layer 1 ((null)): (768, 28, 1, 1): [-0.023491, -0.206194, -0.398334, -0.282982, 0.153860, 0.012939, -0.238999, 0.007868, ]
layer 2 ((null)): (768, 28, 1, 1): [0.041917, -0.297606, -0.175895, -0.050014, 0.410508, -0.295696, -0.162900, -0.047751, ]
layer 3 ((null)): (768, 28, 1, 1): [0.398582, -0.544272, -0.449548, -0.362162, -0.016481, -0.298436, -0.270690, 0.128424, ]
layer 4 ((null)): (768, 28, 1, 1): [0.029922, -0.695409, -0.360548, -0.558251, -0.263455, 0.215809, -0.182401, 0.253034, ]
layer 5 ((null)): (768, 28, 1, 1): [-0.054402, -1.153406, -0.400031, -1.121195, -0.313926, 0.313935, -0.052323, 0.191543, ]
layer 6 ((null)): (768, 28, 1, 1): [-0.066963, -1.067330, -0.822638, -0.149793, 0.275002, 0.624978, 0.201073, 0.321359, ]
layer 7 ((null)): (768, 28, 1, 1): [-0.080141, -0.717065, -0.700867, -0.263565, 0.053924, 0.326026, 0.028566, -0.074821, ]
layer 8 ((null)): (768, 28, 1, 1): [-0.475643, -0.296468, -0.510170, -0.120171, -0.031455, 0.335114, 0.149033, -0.161411, ]
layer 9 ((null)): (768, 28, 1, 1): [-0.885564, -0.398697, -0.011274, -0.073710, -0.122119, 0.018845, -0.109064, 0.572911, ]
layer 10 ((null)): (768, 28, 1, 1): [-0.912256, -0.249506, -0.396818, -0.539064, -0.100314, 0.268732, -0.383853, 1.290616, ]
layer 11 ((null)): (768, 28, 1, 1): [-0.994747, -1.039199, -0.707213, -0.711275, 0.478481, -0.954669, 0.581308, 0.648322, ]
    classify: 6: 0: -0.119204 1: 0.243804 2: 0.135470 3: -0.045005 4: 0.122111 5: -0.249733 
        */

       // i fixed the cpp linear, because it gave different answers. that was from some mismatch stuff
       //     have to keep my eye open, but now i get the same answer
       // i followed the python side of layers, that seems fine too, little off,
       //     also have to keep my eye open
       // but the pooler, is where it is really different.

        

        // classifier
        {
            // inpL = ggml_norm(ctx0, tensor);
            // inpL = my_tensor;

            printf("weights %d %d: ", model.cls_w->ne[0], model.cls_w->ne[1]);
            for (int i = 0; i < 8; i++)
            {
                printf("%f ", ggml_get_f32_1d(model.cls_w, i));
            }
            printf("\n");

            // printf("biases %d: ", model.cls_b->ne[0]);
            // for (int i = 0; i < 8; i++)
            // {
            //     printf("%f ", ggml_get_f32_1d(model.cls_b, i));
            // }
            // printf("\n");

            printf("imported %d %d: ", inpL->ne[0], inpL->ne[1]);
            for (int i = 0; i < 8; i++)
            {
                printf("%f ", ggml_get_f32_1d(inpL, i));
            }
            printf("\n");

            struct ggml_tensor *something = ggml_mul_mat(ctx0,
                    model.cls_w,
                    test);
            printf("mul %d %d: ", something->ne[0], something->ne[1]);
            for (int j = 0; j < something->ne[0]; j++)
            {
                printf("%f ", ggml_get_f32_1d(something, j));
            }
            printf("\n");

            test = ggml_add(ctx0,
                            something,
                            model.cls_b);
                            // 0: -0.010851 1: -0.001317 2: 0.003120 3: 0.021437 4: 0.007453 5: 0.004738
                            // 0: -0.119204 1: 0.243804 2: 0.135470 3: -0.045005 4: 0.122111 5: -0.249733
                            /*
                            imported 768: -0.297266 -0.506810 -0.935216 0.182061 0.797689 -0.491853 -0.371589 0.416660 
mul 6 1: -0.119124 0.243825 0.135329 -0.045008 0.122171 -0.249742 
classify: 6: 0: -0.119204 1: 0.243804 2: 0.135470 3: -0.045005 4: 0.122111 5: -0.249733 
imported 768: 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 
mul 6 1: -0.075254 0.212819 0.359390 0.994326 0.021527 -1.018812 
classify: 6: 0: -0.075334 1: 0.212798 2: 0.359530 3: 0.994329 4: 0.021467 5: -1.018803 
                            */
            
            printf("classify: %d: ", test->ne[0]);
            float *my_outs = ggml_get_data_f32(test);

            // get the index of the max value
            int max_index = 0;
            float max_value = my_outs[0];
            for (size_t j = 1; j < model.cls_b->ne[0]; j++)
            {
                if (my_outs[j] > max_value)
                {
                    max_value = my_outs[j];
                    max_index = j;
                }
            }

            // print the output
            for (size_t j = 0; j < model.cls_b->ne[0]; j++)
            {
                // use ascii escape sequences to make this a different color
                // if its the max, also make it bold
                if (j == max_index) {
                    printf("\033[1;32m");
                } else {
                    printf("\033[0;32m");
                }

                printf("%d: %f ", j, my_outs[j]);
                // reset the color
                printf("\033[0m");
            }
            printf("\n");
        }

        // normalizer
        ggml_tensor *length = ggml_sqrt(ctx0,
                                        ggml_sum(ctx0, ggml_sqr(ctx0, test)));
        printf("%f", ggml_get_data_f32(ggml_div(ctx0, ggml_new_f32(ctx0, 1.0f), length))[0]);
        test = ggml_scale(ctx0, test, ggml_get_data_f32(ggml_div(ctx0, ggml_new_f32(ctx0, 1.0f), length))[0]);
        ggml_tensor *output = test;
        // ggml_tensor *output = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 768);
        // ggml_set_f32(output, 0.5f);
        // run the computation
        // ggml_build_forward_expand(&gf, output);
        
        // print the output layer
        // printf("output layer:\n");
        float *outs = ggml_get_data_f32(output);
        // print the output
        printf("%d: ", output->ne[0]);
        for (size_t j = 0; j < 8; j++)
        {
            printf("%f ", outs[j]);
        }
        printf("...\n");

        
        // ggml_graph_compute_with_ctx(ctx0, &gf, n_threads);


        // float *dat = ggml_get_data_f32(output);
        // pretty_print_tensor(dat, output->ne, output->nb, output->n_dims - 1, "");

        #ifdef GGML_PERF
            // print timing information per ggml operation (for debugging purposes)
            // requires GGML_PERF to be defined
            ggml_graph_print(&gf);
        #endif

        if (!mem_req_mode) {
            memcpy(batch_embeddings[ba], (float *)ggml_get_data(output), sizeof(float) * n_embd);
        } else {
            mem_per_token = ggml_used_mem(ctx0) / N;

            // printf("used_mem = %zu KB \n", ggml_used_mem(ctx0) / 1024);
            // printf("mem_per_token = %zu KB \n", mem_per_token / 1024);
        }

        ggml_free(ctx0);
        printf("free\n");
}

void bert_encode(
    struct bert_ctx *ctx,
    int32_t n_threads,
    const char *texts,
    float *embeddings)
{
    bert_encode_batch(ctx, n_threads, 1, 1, &texts, &embeddings);
}

void bert_encode_batch(
    struct bert_ctx *ctx,
    int32_t n_threads,
    int32_t n_batch_size,
    int32_t n_inputs,
    const char ** texts,
    float **embeddings)
{
    // TODO: Disable batching for now
    n_batch_size = 1;
    /*
    if (n_batch_size > n_inputs) {
        n_batch_size = n_inputs;
    }
    if (n_batch_size > ctx->max_batch_n) {
        bert_resize_ctx(ctx, n_batch_size);
        n_batch_size = ctx->max_batch_n;
    }
    */

    int32_t N = bert_n_max_tokens(ctx);

    std::vector<bert_vocab_id> buf_tokens;
    // Most of this buffer will be unused in typical case where inputs are not that long.
    buf_tokens.resize(N * n_inputs);
    std::vector<int32_t> n_tokens = std::vector<int32_t>(n_inputs);
    std::vector<bert_vocab_id*> unsorted_tokens(n_inputs);
    bert_vocab_id* it_tokens = buf_tokens.data();
    for (int i = 0; i < n_inputs; i++) {
        unsorted_tokens[i] = it_tokens;
        bert_tokenize(ctx, texts[i], it_tokens, &n_tokens[i], N);
        it_tokens += n_tokens[i];
    }

    if (n_batch_size == n_inputs) {
        bert_eval_batch(ctx, n_threads, n_batch_size, unsorted_tokens.data(), n_tokens.data(), embeddings);
    } else {
        // sort the inputs by tokenized length, batch and eval

        std::vector<int> indices;
        indices.reserve(n_inputs);
        for (int i = 0; i < n_inputs; i++)
        {
            indices.push_back(i);
        }

        std::vector<int32_t> sorted_n_tokens = std::vector<int32_t>(n_inputs);

        std::vector<bert_vocab_id *> sorted_tokens(n_inputs);

        std::sort(indices.begin(), indices.end(), [&](int a, int b)
                  { return n_tokens[a] < n_tokens[b]; });

        std::vector<float *> sorted_embeddings(n_inputs);
        memcpy(sorted_embeddings.data(), embeddings, n_inputs * sizeof(float *));

        for (int i = 0; i < n_inputs; i++) {
            sorted_embeddings[i] = embeddings[indices[i]];
            sorted_tokens[i] = unsorted_tokens[indices[i]];
            sorted_n_tokens[i] = n_tokens[indices[i]];
        }

        for (int i = 0; i < n_inputs; i += n_batch_size)
        {
            if (i + n_batch_size > n_inputs) {
                n_batch_size = n_inputs - i;
            }
            bert_eval_batch(ctx, n_threads, n_batch_size, &sorted_tokens[i], &sorted_n_tokens[i], &sorted_embeddings[i]);
        }
    }
}

void bert_test(bert_ctx * ctx) {
    const bert_model &model = ctx->model;
    const auto &hparams = model.hparams;

        const int n_embd = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_max_tokens = hparams.n_max_tokens;
        const int n_head = hparams.n_head;

        const int d_head = n_embd / n_head;

        std::vector<float> result;

        auto & mem_per_token = ctx->mem_per_token;
        auto & buf_compute   = ctx->buf_compute;

        struct ggml_init_params params = {
            .mem_size = buf_compute.size,
            .mem_buffer = buf_compute.data,
            .no_alloc = false,
        };
        printf("start\n");

        struct ggml_context *ctx0 = ggml_init(params);
        printf("ggml_init done\n");

    ggml_tensor *t1 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 8);
    for (int i = 0; i < 8; i++)
    {
        ggml_set_i32_1d(t1, i, 1);
    }
    ggml_tensor *t2 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 8);
    for (int i = 0; i < 8; i++)
    {
        ggml_set_i32_1d(t2, i, 2);
    }
    ggml_tensor *t3 = ggml_get_rows(ctx0, t1, t2);
    printf("embedding: (%d, %d, %d, %d)\n", t3->ne[0], t3->ne[1], t3->ne[2], t3->ne[3]);
    for (size_t j = 0; j < 8; j++)
    {
        printf("%f, ", ggml_get_f32_nd(t3, j, j, 0, 0));
    }
    printf("...\n");
    printf("================= test done :D ===================\n");
}
