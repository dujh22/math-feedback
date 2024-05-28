@echo off                                       :: 关闭命令回显，不显示执行的命令。

SET project_path=F://code//github//ChatGLM-MathV2//  :: 设置项目路径变量。
SET num_parallel_process=1                        :: 设置并行处理过程的数量。
cd %project_path%                                 :: 切换到项目路径。

echo Starting the data preprocessing pipeline...  :: 打印开始数据预处理的消息。

cd %project_path%utils                            :: 切换到项目的utils目录。
SET dataset=math_shepherd                         :: 设置数据集名称变量。
SET has_label=hasset                              :: 设置是否有标签的变量。
SET has_response=has                              :: 设置是否有响应的变量。
SET input_file_path=%project_path%raw_data//peiyi9979_Math_Shepherd//math-shepherd.jsonl  :: 设置输入文件路径。
SET num_points=10                                 :: 设置处理数据点的数量。
SET new_folder_suffix=math_shepherd_test_data%num_points%  :: 设置新文件夹后缀。
SET language=en                                   :: 设置语言为英语。
SET output_file_path=%project_path%data//%new_folder_suffix%//%new_folder_suffix%.jsonl  :: 设置输出文件路径。
python data_preprocessing.py %input_file_path% %output_file_path% %num_points% %language% %dataset% %has_label% %has_response%  :: 运行数据预处理脚本。

echo Data preprocessing completed.                :: 打印数据预处理完成的消息。

echo Starting backend generation...               :: 打印开始后端生成的消息。

cd %project_path%shepherd_prm                     :: 切换到shepherd_prm目录。
SET prompt_template_path=%project_path%shepherd_prm//templates//criticllm_math_template.txt  :: 设置提示模板路径。
SET prompt_key=question                           :: 设置问题键。
SET response_key=response                         :: 设置响应键。
SET reference_key=answer                          :: 设置参考答案键。

SET backbone=tgi                                  :: 设置背骨网络为tgi。
SET input_file_path=%project_path%data//%new_folder_suffix%//%new_folder_suffix%.jsonl  :: 更新输入文件路径。
SET mode=response                                 :: 设置模式为响应。
SET num_process=%num_parallel_process%            :: 设置处理过程数。
SET url=http://172.18.64.8:8080/generate          :: 设置API URL。

python query_api.py --input_file %input_file_path% --prompt_template %prompt_template_path% --prompt_key %prompt_key% --response_key %response_key% --reference_key %reference_key% --backbone %backbone% --mode %mode% --num_process %num_process% --url %url%  :: 运行查询API脚本。

echo Backend generation completed.                :: 打印后端生成完成的消息。
echo Starting backend scoring...                  :: 打印开始后端评分的消息。

SET backbone=tgi                                  :: 重设背骨网络为tgi。
SET input_file_path=%project_path%data//%new_folder_suffix%//%new_folder_suffix%_tgi.jsonl  :: 更新输入文件路径。
SET mode=critic                                   :: 设置模式为批评。
SET num_process=%num_parallel_process%            :: 设置处理过程数。
SET url=http://172.18.64.55:8080/generate         :: 更新API URL。

python query_api.py --input_file %input_file_path% --prompt_template %prompt_template_path% --prompt_key %prompt_key% --response_key %response_key% --reference_key %reference_key% --backbone %backbone% --mode %mode% --num_process %num_process% --url %url%  :: 再次运行查询API脚本。

echo Backend scoring completed.                   :: 打印后端评分完成的消息。
echo Starting forward process path prediction...  :: 打印开始前进过程路径预测的消息。

SET prompt_template_path=%project_path%shepherd_prm//templates//criticllm_math_template.txt  :: 重设提示模板路径。
SET prompt_key=question                           :: 重设问题键。
SET response_key=response                         :: 重设响应键。
SET reference_key=answer                          :: 重设参考答案键。
SET process_response_key=generated_paths          :: 设置过程响应键。
SET reference_answer_key=answer                    :: 重设参考答案键。

SET backbone=tgi                                  :: 重设背骨网络为tgi。
SET input_file_path=%project_path%data//%new_folder_suffix%//%new_folder_suffix%_tgi_math_critic.jsonl  :: 更新输入文件路径。
SET mode=generation                               :: 设置模式为生成。
SET num_process=%num_parallel_process%            :: 重设处理过程数。
SET url=http://172.18.64.8:8080/generate          :: 重设API URL。

python prm_evaluate_process.py --input_file %input_file_path% --prompt_template %prompt_template_path% --prompt_key %prompt_key% --response_key %response_key% --reference_key %reference_key% --process_response_key %process_response_key% --reference_answer_key %reference_answer_key% --backbone %backbone% --mode %mode% --num_process %num_process% --url %url%  :: 运行过程评价脚本。

echo Forward process path prediction completed.   :: 打印前进过程路径预测完成的消息。
echo Starting forward process path evaluation...  :: 打印开始前进过程路径评价的消息。

SET backbone=tgi                                  :: 重设背骨网络为tgi。
SET input_file_path=%project_path%data//%new_folder_suffix%//%new_folder_suffix%_tgi_math_critic_path.jsonl  :: 更新输入文件路径。
SET mode=critic                                   :: 重设模式为批评。
SET num_process=%num_parallel_process%            :: 重设处理过程数。
SET url=http://172.18.64.55:8080/generate         :: 重设API URL。

python prm_evaluate_process.py --input_file %input_file_path% --prompt_template %prompt_template_path% --prompt_key %prompt_key% --response_key %response_key% --reference_key %reference_key% --process_response_key %process_response_key% --reference_answer_key %reference_answer_key% --backbone %backbone% --mode %mode% --num_process %num_process% --url %url%  :: 再次运行过程评价脚本。

echo Forward process path evaluation completed.   :: 打印前进过程路径评价完成的消息。
echo Calculating accuracy...                      :: 打印开始计算准确性的消息。

cd %project_path%                                 :: 切换到项目路径。

SET file_path=%project_path%data//%new_folder_suffix%//%new_folder_suffix%_tgi_math_critic_path_math_critic2.jsonl  :: 设置文件路径。
SET output_file_path=%project_path%data//%new_folder_suffix%//%new_folder_suffix%_tgi_math_critic_path_math_critic2_statistics.csv  :: 设置输出文件路径。

python Check3_CalculatePathPredictAccuracy.py %file_path% %output_file_path%  :: 运行计算路径预测准确性脚本。

echo Accuracy calculation completed.              :: 打印准确性计算完成的消息。
echo Starting forward automatic labeling...       :: 打印开始前进自动标记的消息。

cd %project_path%utils                            :: 切换回utils目录。

SET input_file_path=%project_path%data//%new_folder_suffix%//%new_folder_suffix%_tgi_math_critic_path_math_critic2.jsonl  :: 更新输入文件路径。
SET output_file_path=%project_path%data//%new_folder_suffix%//front//%new_folder_suffix%.jsonl  :: 设置输出文件路径。
python turn_response_and_solution.py %input_file_path% %output_file_path%  :: 运行转换响应和解决方案脚本。

cd %project_path%                                 :: 切换回项目根目录。

SET source_folder=%project_path%data//%new_folder_suffix%//front  :: 设置源文件夹路径。
SET target_folder=%project_path%data//%new_folder_suffix%//front_step1  :: 设置目标文件夹路径。
python Step1_SplitByRow.py %source_folder% %target_folder%  :: 运行按行分割数据的脚本。

echo Forward automatic labeling finished Step1_SplitByRow.  :: 打印完成第一步分割的消息。

SET source_folder=%project_path%data//%new_folder_suffix%//front_step1  :: 更新源文件夹路径。
SET target_folder=%project_path%data//%new_folder_suffix%//front_step2  :: 更新目标文件夹路径。
python Step2_IsCalculationOrReasoning.py %source_folder% %target_folder%  :: 运行判断计算或推理的脚本。

echo Forward automatic labeling finished Step2_IsCalculationOrReasoning.  :: 打印完成第二步判断的消息。

SET source_folder=%project_path%data//%new_folder_suffix%//front_step2  :: 更新源文件夹路径。
SET target_folder=%project_path%data//%new_folder_suffix%//front_step3  :: 更新目标文件夹路径。
SET backbone=tgi                                  :: 重设背骨网络为tgi。
SET url=http://172.18.64.8:8080/generate          :: 重设API URL。
SET max_words=%num_parallel_process%              :: 设置最大词数变量。
python Step3_JudgmentStepCalculatedCorrectly.py %source_folder% %target_folder% %max_words% %backbone% %url%  :: 运行判断步骤计算正确性的脚本。

echo Forward automatic labeling finished Step3_JudgmentStepCalculatedCorrectly.  :: 打印完成第三步判断的消息。

SET source_folder=%project_path%data//%new_folder_suffix%//front_step3  :: 更新源文件夹路径。
SET target_folder=%project_path%data//%new_folder_suffix%//front_step4  :: 更新目标文件夹路径。
SET max_words=%num_parallel_process%              :: 重设最大词数变量。
SET backbone1=tgi                                 :: 设置第一个背骨网络为tgi。
SET url1=http://172.18.64.8:8080/generate         :: 设置第一个API URL。
SET backbone2=tgi                                 :: 设置第二个背骨网络为tgi。
SET url2=http://172.18.64.55:8080/generate        :: 设置第二个API URL。
python Step4_JudgmentStepReasoningCorrectly.py %source_folder% %target_folder% %max_words% %backbone1% %url1% %backbone2% %url2%  :: 运行判断推理正确性的脚本。

echo Forward automatic labeling finished Step4_JudgmentStepReasoningCorrectly.  :: 打印完成第四步判断的消息。

echo Forward automatic labeling completed.        :: 打印前进自动标记完成的消息。

echo Forward process path evaluation completed.   :: 打印前进过程路径评价完成的消息。

echo Calculating accuracy2...                      :: 打印开始计算准确性的消息。

SET file_path=%project_path%data//%new_folder_suffix%//front_step4//%new_folder_suffix%.jsonl  :: 设置文件路径。
SET output_file_path=%project_path%data//%new_folder_suffix%//front_step4//%new_folder_suffix%_2.csv  :: 设置输出文件路径。

python Check3_CalculatePathPredictAccuracy.py %file_path% %output_file_path%  :: 运行计算路径预测准确性脚本。

echo Accuracy2 calculation completed.              :: 打印准确性计算完成的消息。

echo All steps completed successfully!            :: 打印所有步骤成功完成的消息。

pause                                            :: 暂停执行，等待用户按任意键继续。
