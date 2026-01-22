Text-to-SQL 논문 계보
1. RASAT: Integrating Relational Structures into Pretrained Seq2Seq Model for Text-to-SQL
2. DIN-SQL : Decomposed In-Context Learning of Text-to-SQL with Self-Correction
3. C3: Zero-shot Text-to-SQL with ChatGPT
4. ACT-SQL: In-Context Learning for Text-to-SQL with Automatically-Generated Chain-of-Thought
5. MAC-SQL: A Multi-Agent Collaborative Framework for Text-to-SQL
   - Selector   : 필요있는 스키마만 남기기
   - Decomposer : 태스크를 작게 쪼개어 단계별로 풀면서 SQL 구성(안정성)
   - Refiner    : SQL 실행해보고 오류 있으면, 피드백을 바탕으로 SQL 을 수정
6. PET-SQL: A Prompt-Enhanced Two-Round Refinement of Text-to-SQL with Cross-consistency
7. Decomposition for Enhancing Attention: Improving LLM-based Text-to-SQL through Workflow Paradigm
8. CoE-SQL: In-Context Learning for Multi-Turn Text-to-SQL with Chain-of-Editions
9. CHESS: Contextual Harnessing for Efficient SQL Synthesis
10. Before Generation, Align it! A Novel and Effective Strategy for Mitigating Hallucinations in Text-to-SQL Generation


아키텍쳐
<img width="748" height="477" alt="image" src="https://github.com/user-attachments/assets/e3ce1bcf-5fa2-4b22-b294-f8855d33af96" />



구현 내용
1. streamlit 기반 ui
2. text2sql agent
3. text2sql mcp server


실행 방법(streamlit ui)
 - streamlit run app.py  => streamlit ui 에서 agent 호출하여 text2sql 로직 수행

실행 방법(mcp server)
 - python server.py

실행 방법(continue mcp tool 등록)
<img width="1371" height="733" alt="image" src="https://github.com/user-attachments/assets/cf85006f-45e4-4da4-9341-1040a7f5ccbf" />

<img width="607" height="344" alt="image" src="https://github.com/user-attachments/assets/a33dfa42-95a5-4f42-8925-d9fbfad2ac80" />

<img width="706" height="750" alt="image" src="https://github.com/user-attachments/assets/39162679-99dc-45f1-a8fe-b56581a29919" />

<img width="708" height="727" alt="image" src="https://github.com/user-attachments/assets/a2d8807a-3279-47fd-91c8-7a8d43005b81" />
