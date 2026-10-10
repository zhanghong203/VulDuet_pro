import json
import re
import time


class AnalysisAgent:
	"""
    Analysis Agent

    功能：
        输入C/C++代码
        ↓
        LLM分析代码
        ↓
        输出用于规则检索的JSON
    """

	def __init__(
			self,
			llm_client,
			model_settings=None,
			retry_times=3
	):
		self.llm = llm_client
		self.model_settings = model_settings
		self.retry_times = retry_times

	####################################################################
	# Prompt
	####################################################################

	def build_prompt(self, code: str):
		prompt = f"""
		You are an expert in C/C++ secure coding and program analysis.

		Your task is NOT to determine whether the code is vulnerable.
		
		Instead, generate a retrieval description that summarizes the security-relevant semantics of the code. The description will be embedded and used to retrieve the most relevant SEI CERT C secure coding rules.
		
		Requirements:

			1. Write ONE coherent paragraph.
			2. Describe what the code does from a security perspective.
			3. Focus on memory operations, pointer usage, array access, integer operations, resource management, external input handling, buffer manipulation, synchronization, and other security-relevant behaviors.
			4. Describe behaviors only. Do NOT determine whether the code is vulnerable.
			5. Do NOT mention CWE, CVE, vulnerability names, or exploitability.
			6. Do NOT output bullet points, numbered lists, or keywords.
			7. The writing style should resemble the natural language used in the SEI CERT C Coding Standard.

		Example
		
		Input Code:
		
		```c
		char *copy(char *src)
		{{
			char buf[32];
			strcpy(buf, src);
			return strdup(buf);
		}}
		Output:
		
		{{
		"security_description": "The function receives a character string from an external source, copies the data into a fixed-size local buffer, and allocates new heap memory to duplicate the resulting string. The implementation performs pointer-based string manipulation and memory allocation while relying on buffer operations whose correctness depends on proper boundary management and safe handling of externally supplied data."
		}}

		Now analyze the following code.

		Code:
		{code}
		Return ONLY valid JSON in the following format:

		{{
		"security_description": ""
		}}
		"""
		return prompt

	####################################################################
	# 调用LLM
	####################################################################

	def analyze(self, code: str):

		prompt = self.build_prompt(code)

		response = None

		for attempt in range(self.retry_times):

			try:

				response = self.llm.generate(prompt)

				if response is None:
					raise Exception("LLM returns None.")

				result = self.parse_response(response)

				return result

			except Exception as e:

				print(
					f"[AnalysisAgent] Retry {attempt + 1}: {e}"
				)

				time.sleep(1)

		raise RuntimeError(
			"Analysis Agent Failed."
		)

	####################################################################
	# 解析LLM输出
	####################################################################

	def parse_response(self, response):

		response = response.strip()

		# 去掉markdown
		response = re.sub(r"```json", "", response)
		response = re.sub(r"```", "", response)

		response = response.strip()

		try:

			result = json.loads(response)

		except Exception:

			# 尝试提取JSON
			match = re.search(
				r"\{.*\}",
				response,
				re.S
			)

			if match is None:
				raise Exception(
					"Cannot parse JSON from LLM output."
				)

			result = json.loads(match.group())

		return result

