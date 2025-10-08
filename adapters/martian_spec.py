@pytest.mark.slow
@pytest.mark.asyncio
async def test_single_java_pipeline_with_live_model(java_sandbox_client, java_problem):
    """
    Tests the full pipeline for a single problem, including a live call to an LLM.
    This is an integration test and should be run selectively.
    """
    language="java"
    prompt = java_problem["prompt"]
    base_instruction = utils.get_base_instruction(language)
    lang_instructions = utils.get_lang_specific_prompt_instructions(language=language)
    final_instuction = base_instruction + "\n" + lang_instructions

    final_prompt = f"Provide a complete implementation for this {language} function:\n\n```\n{prompt}\n```\n\n{final_instuction}"  # noqa: E501

    messages = [
        openai.types.chat.ChatCompletionSystemMessageParam(role="system", content=f"You are a skilled {language} programmer."),  # noqa: E501
        openai.types.chat.ChatCompletionUserMessageParam(role="user", content=final_prompt),
    ]

    api_client = openai.AsyncOpenAI(base_url="https://api.withmartian.com/v1")
    response = await api_client.chat.completions.create(
        model="openai/gpt-4.1-nano",
        messages=messages,
        temperature=0.0
    )

    model_solution = clean_and_validate_code(request=None, completion=response, language=language)

    ## adjust this for java file. ##
    code_to_run = utils.assemble_java_file(problem=java_problem, model_generated_code=model_solution)
    print("===CODE TO RUN===\n", code_to_run)
    request_obj = code_sandbox_types.CodeRequest(code=code_to_run, timeout_s=60)

    result = await java_sandbox_client.run_code(req_data=request_obj, programming_language=language)
    print("===RESULT===\n", result)

    passed = (result.stderr == "" and result.returncode == 0)
    assert passed, f"Model-generated solution failed with stderr:\n{result.stderr}"