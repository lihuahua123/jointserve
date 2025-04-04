import time

from vllm import LLM, SamplingParams
from vllm.utils import FlexibleArgumentParser
import jsonlines

PROMPT = "You are a helpful assistant in recognizes the content of tables in markdown format. Here is a table as fellows. You need to answer my question about the table.\n# Table\n|Opening|Opening|Sl. No.|Film|Cast|Director|Music Director|Notes|\n|----|----|----|----|----|----|----|----|\n|J A N|9|1|Agni Pushpam|Jayabharathi, Kamalahasan|Jeassy|M. K. Arjunan||\n|J A N|16|2|Priyamvada|Mohan Sharma, Lakshmi, KPAC Lalitha|K. S. Sethumadhavan|V. Dakshinamoorthy||\n|J A N|23|3|Yakshagaanam|Madhu, Sheela|Sheela|M. S. Viswanathan||\n|J A N|30|4|Paalkkadal|Sheela, Sharada|T. K. Prasad|A. T. Ummer||\n|F E B|5|5|Amma|Madhu, Srividya|M. Krishnan Nair|M. K. Arjunan||\n|F E B|13|6|Appooppan|Thikkurissi Sukumaran Nair, Kamal Haasan|P. Bhaskaran|M. S. Baburaj||\n|F E B|20|7|Srishti|Chowalloor Krishnankutty, Ravi Alummoodu|K. T. Muhammad|M. S. Baburaj||\n|F E B|20|8|Vanadevatha|Prem Nazir, Madhubala|Yusufali Kechery|G. Devarajan||\n|F E B|27|9|Samasya|Madhu, Kamalahaasan|K. Thankappan|Shyam||\n|F E B|27|10|Yudhabhoomi|K. P. Ummer, Vidhubala|Crossbelt Mani|R. K. Shekhar||\n|M A R|5|11|Seemantha Puthran|Prem Nazir, Jayabharathi|A. B. Raj|M. K. Arjunan||\n|M A R|12|12|Swapnadanam|Rani Chandra, Dr. Mohandas|K. G. George|Bhaskar Chandavarkar||\n|M A R|19|13|Thulavarsham|Prem Nazir, sreedevi, Sudheer|N. Sankaran Nair|V. Dakshinamoorthy||\n|M A R|20|14|Aruthu|Kaviyoor Ponnamma, Kamalahasan|Ravi|G. Devarajan||\n|M A R|26|15|Swimming Pool|Kamal Haasan, M. G. Soman|J. Sasikumar|M. K. Arjunan||\n\n# Question\nWhat' s the content in the (1,1) cells\n"  # noqa: E501
warm_prompts = []
with open("sharegpt_gpt4.jsonl", "r+", encoding="utf8") as f:
    for item in jsonlines.Reader(f):
        warm_prompts.append(item["conversations"][0]["value"])


def test_prefix(llm=None, sampling_params=None, prompts=None,print_time=False):
    start_time = time.time()

    llm.generate(prompts, sampling_params=sampling_params)

    end_time = time.time()
    if print_time:
        # file_path = 'log1'
        # with open(file_path, 'a', encoding='utf-8') as file:
        #     file.write(f"cost time {end_time - start_time}")
        print(f"cost time {end_time - start_time}")


def main(args):
    llm = LLM(model=args.model,
              tokenizer_mode='auto',
              trust_remote_code=True,
              enforce_eager=False,
              use_v2_block_manager=args.use_v2_block_manager,
              tensor_parallel_size=args.tensor_parallel_size,
              enable_prefix_caching=args.enable_prefix_caching)

    num_prompts = 1
    prompts = [PROMPT] * num_prompts
    sampling_params = SamplingParams(temperature=0, max_tokens=args.output_len)

    #print("------warm up------")
    for i in range(args.warm_up_num):
        test_prefix(
            llm=llm,
            prompts=[warm_prompts[i]],
            sampling_params=sampling_params,
        )

    print("------start generating------")
    begin_time=time.time()
    for i in range(args.warm_up_num,args.warm_up_num+100):
        test_prefix(
            llm=llm,
            prompts=[warm_prompts[i]],
            sampling_params=sampling_params,
            print_time =False
        )
    end_time=time.time()
    print(end_time-begin_time)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description='Benchmark the performance with or without automatic '
        'prefix caching.')
    parser.add_argument('--model',
                        type=str,
                        default='baichuan-inc/Baichuan2-13B-Chat')
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--output-len', type=int, default=10)
    parser.add_argument('--enable-prefix-caching',
                        action='store_true',
                        help='enable prefix caching')
    parser.add_argument('--use-v2-block-manager',
                        action='store_true',
                        help='Use BlockSpaceMangerV2')
    parser.add_argument('--warm-up-num',
                         type=int, default=1)
    args = parser.parse_args()
    main(args)
