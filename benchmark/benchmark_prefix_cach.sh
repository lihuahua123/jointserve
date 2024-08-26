
# 定义数组
my_array=(10 50 100 150 200)

# 使用数组元素循环赋值给变量 num
for num in "${my_array[@]}"
do
  echo $num
  python benchmark_prefix_caching.py --model /hy-tmp/ --enable-prefix-caching --use-v2-block-manager --warm-up-num $num
done
