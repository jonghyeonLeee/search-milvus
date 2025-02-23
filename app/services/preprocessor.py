from bs4 import BeautifulSoup
from typing import Dict, Any, List
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords

from pygments.lexers import guess_lexer

nltk.download('stopwords')
nltk.download('punkt')


class DocumentPreprocessor:
    def __init__(self):
        """
        문서 전처리기 초기화
        - tfidf: 키워드 추출을 위한 TF-IDF 벡터라이저
        """
        self.tfidf = TfidfVectorizer(
            max_features=100,
            stop_words=stopwords.words('english') + ['만', '및', '등', '를', '을', '이', '가']
        )

    def _clean_html(self, html_text: str) -> str:
        """
        HTML 문서 정제
        1. BeautifulSoup으로 HTML 파싱
        2. <pre>, <code> 태그 내용은 코드 블록으로 보존
        3. 나머지 HTML 태그 제거
        4. 불필요한 공백 제거

        Args:
            html_text (str): 원본 HTML 텍스트

        Returns:
            str: 정제된 텍스트
        """
        soup = BeautifulSoup(html_text, 'html.parser')

        # 코드 블록 보존
        for code in soup.find_all(['pre', 'code']):
            code.replace_with(f"```\n{code.get_text()}\n```")

        clean_text = soup.get_text()
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        return clean_text

    def _analyze_document_type(self, text: str, code_blocks: List[tuple[str, str]]) -> str:
        """
        문서 타입 분석
        - 코드 블록 수와 텍스트 내용을 기반으로 문서 성격 파악

        Args:
            text (str): 정제된 텍스트
            code_blocks (List[tuple]): 추출된 코드 블록들

        Returns:
            str: 문서 타입 설명
        """
        code_ratio = len(''.join(code for _, code in code_blocks)) / len(text)

        if code_ratio > 0.5:
            return "코드 중심의 기술 문서"
        elif "가이드" in text or "방법" in text:
            return "사용자 가이드 문서"
        elif "오류" in text or "에러" in text or "해결" in text:
            return "문제 해결(트러블슈팅) 문서"
        else:
            return "일반 기술 문서"

    # app/services/preprocessor.py의 _summarize_text 메서드 수정
    def _summarize_text(self, text: str, code_blocks: List[tuple[str, str]]) -> str:
        """
        문서 내용 분석 및 구조화된 요약 생성

        1. 코드 블록 제거 후 본문 추출
        2. 문서 타입 분석
        3. 포함된 프로그래밍 언어 확인
        4. 첫 몇 문장을 기반으로 한 요약 생성

        Args:
            text (str): 정제된 텍스트
            code_blocks (List[tuple]): 추출된 코드 블록들

        Returns:
            str: 구조화된 요약문
        """
        # 코드 블록 제거
        text_without_code = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

        try:
            # 문서 타입 분석
            doc_type = self._analyze_document_type(text, code_blocks)

            # 프로그래밍 언어 목록
            languages = list(set(lang for lang, _ in code_blocks if lang != "unknown"))

            # 첫 3문장을 요약문으로 사용
            sentences = text_without_code.split('.')[:3]
            base_summary = '. '.join(sentences).strip()

            # 구조화된 요약문 생성
            structured_summary = f"""
            문서 유형: {doc_type}

            포함된 프로그래밍 언어: {', '.join(languages) if languages else '없음'}

            주요 내용: {base_summary}

            코드 예제: {len(code_blocks)}개의 코드 블록 포함
            """

            return structured_summary.strip()

        except Exception as e:
            print(f"요약 생성 중 오류 발생: {str(e)}")
            sentences = text_without_code.split('.')[:3]
            return '. '.join(sentences).strip()

    def _extract_code_blocks(self, text: str) -> List[tuple[str, str]]:
        """
        코드 블록 추출 및 언어 감지
        1. 마크다운 코드 블록 패턴 매칭
        2. 언어 태그 확인
        3. 코드 컨텍스트 기반 언어 추측

        Args:
            text (str): 정제된 텍스트

        Returns:
            List[tuple]: (언어, 코드) 튜플 리스트
        """
        pattern = r"```(\w+)?\n(.*?)\n```"
        matches = re.findall(pattern, text, re.DOTALL)
        code_blocks = []

        for lang, code in matches:
            # 코드 컨텍스트 분석
            context_before = text.split(f"```{lang}\n{code}\n```")[0][-100:]
            lang = self._detect_language_from_context(code, context_before) if not lang else lang
            code_blocks.append((lang, code))

        return code_blocks

    def _detect_language_from_context(self, code: str, context: str) -> str:
        """
        코드와 주변 컨텍스트를 기반으로 프로그래밍 언어 감지

        Args:
            code (str): 코드 내용
            context (str): 코드 블록 이전의 컨텍스트

        Returns:
            str: 감지된 프로그래밍 언어
        """
        # 컨텍스트에서 언어 관련 키워드 찾기
        context_lower = context.lower()
        if "자바" in context_lower or "java" in context_lower:
            return "java"
        elif "파이썬" in context_lower or "python" in context_lower:
            return "python"
        elif "자바스크립트" in context_lower or "javascript" in context_lower:
            return "javascript"

        try:
            lexer = guess_lexer(code)
            return lexer.name.lower()
        except Exception:
            return "unknown"

    def _extract_tags(self, text: str) -> List[str]:
        """
        TF-IDF 기반 키워드 추출
        1. 코드 블록 제거
        2. TF-IDF 적용
        3. 상위 키워드 추출

        Args:
            text (str): 정제된 텍스트

        Returns:
            List[str]: 추출된 키워드 리스트
        """
        # 코드 블록 제거
        text_without_code = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

        # TF-IDF 적용
        tfidf_matrix = self.tfidf.fit_transform([text_without_code])
        feature_names = self.tfidf.get_feature_names_out()

        # 상위 키워드 추출
        scores = zip(feature_names, tfidf_matrix.toarray()[0])
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

        # 상위 10개 키워드 반환
        return [word for word, score in sorted_scores[:10] if score > 0]

    def _create_enhanced_text(self, text: str, summary: str, tags: List[str]) -> str:
        """
        강화된 텍스트 생성
        - 요약, 태그, 원본 내용을 포함한 구조화된 텍스트 생성

        Args:
            text (str): 원본 텍스트
            summary (str): 요약문
            tags (List[str]): 추출된 태그들

        Returns:
            str: 강화된 텍스트
        """
        return f"""
Summary: {summary}

Tags: {', '.join(tags)}

Original Content:
{text}
"""

    async def process_document(self, text: str) -> Dict[str, Any]:
        """
        문서 전처리 메인 프로세스
        1. HTML 정제
        2. 코드 블록 추출
        3. 문서 분석 및 요약
        4. 메타데이터 생성

        Args:
            text (str): 원본 문서 텍스트

        Returns:
            Dict: 처리된 문서 정보 {
                text: str,          # 정제된 텍스트
                metadata: {         # 문서 메타데이터
                    summary: str,   # 문서 요약
                    code_blocks: {  # 코드 블록 정보
                        language: {
                            count: int,
                            samples: List[str]
                        }
                    },
                    tags: List[str] # 추출된 태그
                },
                original_text: str  # 원본 텍스트
            }
        """
        clean_text = self._clean_html(text)
        code_blocks = self._extract_code_blocks(clean_text)
        summary = self._summarize_text(clean_text, code_blocks)
        tags = self._extract_tags(clean_text)

        # 프로그래밍 언어별 코드 블록 그룹화
        code_by_lang = {}
        for lang, code in code_blocks:
            if lang not in code_by_lang:
                code_by_lang[lang] = []
            code_by_lang[lang].append(code)

        metadata = {
            "summary": summary,
            "code_blocks": {
                lang: {
                    "count": len(codes),
                    "samples": codes
                }
                for lang, codes in code_by_lang.items()
            },
            "tags": tags
        }

        enhanced_text = self._create_enhanced_text(clean_text, summary, tags)

        return {
            "text": enhanced_text,
            "metadata": metadata,
            "original_text": text
        }