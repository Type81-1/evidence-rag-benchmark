from __future__ import annotations

import argparse

from clinical_assistant import answer_question


EXAMPLE_QUESTIONS = {
    "hypertension": "高血压患者为什么有时要长期吃药？有哪些指南或研究依据？",
    "lipids": "体检发现血脂偏高，生活方式干预和药物治疗分别有哪些证据？",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenEvidence 临床证据助手")
    parser.add_argument("question", nargs="?", default=EXAMPLE_QUESTIONS["lipids"])
    parser.add_argument("--offline", action="store_true", help="只使用内置证据，不访问外部 API")
    args = parser.parse_args()
    result = answer_question(args.question, online=not args.offline, use_llm=not args.offline)
    print(result["answer"])
    if result["warnings"]:
        print("\n运行提示：")
        for warning in result["warnings"]:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
