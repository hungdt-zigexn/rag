#!/usr/bin/env ruby
# frozen_string_literal: true

# ===============================================================================
# Query Transform RAG (Phiên bản Ruby)
# ===============================================================================
#
# Query Transform RAG giải quyết vấn đề "Query-Document Mismatch" - khi user query
# và document content có semantic gap, dẫn đến retrieval kém hiệu quả.
#
# Kỹ thuật này thực hiện transformation trên user query để tăng khả năng
# match với relevant documents. Bao gồm các strategies:
#
# 1. **HyDE (Hypothetical Document Embeddings)**: Tạo document giả định
#    từ query, rồi dùng nó để tìm kiếm similar documents thực.
# 2. **Sub-question Decomposition**: Chia query phức tạp thành các
#    sub-questions đơn giản hơn.
# 3. **Step-back Prompting**: Tạo broader, more general questions.
# 4. **Query Rewriting**: Paraphrase query với nhiều cách diễn đạt khác nhau.
# 5. **Multi-Query Fusion**: Kết hợp kết quả từ multiple transformed queries.

require 'net/http'
require 'uri'
require 'json'
require 'matrix'
require 'pdf-reader'

# ===============================================================================
# Thiết lập Môi trường
# ===============================================================================
puts "=== Query Transform RAG ==="

# Cấu hình
OPENAI_API_KEY = ENV['OPENAI_API_KEY']
BASE_URL = 'https://api.studio.nebius.com/v1/'
EMBEDDING_MODEL = 'BAAI/bge-en-icl'
CHAT_MODEL = 'meta-llama/Llama-3.3-70B-Instruct'

# Kiểm tra API key
if OPENAI_API_KEY.nil? || OPENAI_API_KEY.empty?
  puts "Cảnh báo: Biến môi trường OPENAI_API_KEY chưa được đặt"
  exit 1
end

# ===============================================================================
# Hàm API Client
# ===============================================================================

def make_api_request(endpoint, payload)
  # Thực hiện API request đến OpenAI-compatible endpoint.
  uri = URI("#{BASE_URL}#{endpoint}")
  http = Net::HTTP.new(uri.host, uri.port)
  http.use_ssl = true

  request = Net::HTTP::Post.new(uri)
  request['Authorization'] = "Bearer #{OPENAI_API_KEY}"
  request['Content-Type'] = 'application/json'
  request.body = payload.to_json

  response = http.request(request)

  if response.code.to_i == 200
    JSON.parse(response.body)
  else
    puts "Lỗi API: #{response.code} - #{response.body}"
    nil
  end
end

def create_embeddings(text_list, model = EMBEDDING_MODEL)
  # Tạo embeddings cho danh sách văn bản.
  payload = {
    model: model,
    input: text_list
  }

  response = make_api_request('embeddings', payload)
  return [] unless response && response['data']

  response['data'].map { |item| item['embedding'] }
end

def generate_response(system_prompt, user_message, model = CHAT_MODEL)
  # Tạo phản hồi từ mô hình AI.
  payload = {
    model: model,
    temperature: 0.3,  # Lower temperature for more consistent transformations
    messages: [
      { role: 'system', content: system_prompt },
      { role: 'user', content: user_message }
    ]
  }

  response = make_api_request('chat/completions', payload)
  return nil unless response && response['choices'] && !response['choices'].empty?

  response['choices'][0]['message']['content']
end

# ===============================================================================
# Text Processing Utilities
# ===============================================================================

def extract_text_from_pdf(pdf_path)
  # Trích xuất văn bản từ file PDF.
  reader = PDF::Reader.new(pdf_path)
  all_text = ""

  reader.pages.each do |page|
    all_text += page.text + " "
  end

  all_text.strip
end

def chunk_text(text, chunk_size = 1000, overlap = 200)
  # Chia văn bản thành các chunk có overlap.
  chunks = []
  step_size = chunk_size - overlap

  (0...text.length).step(step_size) do |i|
    chunk = text[i, chunk_size]
    chunks << chunk unless chunk.empty?
  end

  chunks
end

# ===============================================================================
# Query Transformation Strategies
# ===============================================================================

def hyde_transform(query, model = CHAT_MODEL)
  # HyDE (Hypothetical Document Embeddings) transformation.
  #
  # Tạo ra một document giả định trả lời câu hỏi, sau đó sử dụng
  # document này để tìm kiếm thay vì dùng trực tiếp query gốc.
  #
  # Ý tưởng: Document embedding thường rich hơn query embedding,
  # nên tạo ra document giả định sẽ tăng quality của embedding.
  #
  # Args:
  #   query (String): Query gốc của user.
  #   model (String): Mô hình LLM để sử dụng.
  #
  # Returns:
  #   String: Hypothetical document được tạo ra.
  system_prompt = <<~HEREDOC
  Bạn là một chuyên gia trong việc tạo ra các document giả định (hypothetical documents)
  để cải thiện tìm kiếm thông tin.

  Nhiệm vụ của bạn: Từ một câu hỏi/query, hãy tạo ra một đoạn văn bản có thể trả lời
  câu hỏi đó. Document này sẽ được dùng làm "cầu nối" để tìm kiếm các document thực sự.

  HƯỚNG DẪN QUAN TRỌNG:
  1. **Comprehensive**: Bao phủ nhiều aspects của question
  2. **Informative**: Chứa details, examples, specific information
  3. **Natural**: Viết như một real document, không như AI response
  4. **Structured**: Có flow logic, organized information
  5. **Relevant**: Focus chính xác vào core của question

  FORMAT: Trả về chỉ hypothetical document, KHÔNG có meta-commentary.

  EXAMPLE INPUT: "What are the benefits of renewable energy?"
  EXAMPLE OUTPUT: "Renewable energy sources provide numerous environmental and economic benefits. Solar and wind power significantly reduce carbon emissions compared to fossil fuels, helping combat climate change. These technologies create jobs in manufacturing, installation, and maintenance sectors. Long-term cost savings are substantial as renewable sources have minimal operating costs after initial investment. Energy independence is enhanced, reducing reliance on volatile fossil fuel markets. Modern renewable systems offer improved reliability and grid stability..."
  HEREDOC

  user_message = "Hãy tạo hypothetical document cho query sau:\n\n#{query}"

  response = generate_response(system_prompt, user_message, model)

  if response
    puts "✓ HyDE Document được tạo (#{response.length} ký tự)"
    puts "Preview: #{response[0..150]}..."
    response
  else
    puts "❌ Không thể tạo HyDE document"
    query  # Fallback to original query
  end
end

def sub_question_decomposition(query, model = CHAT_MODEL)
  # Decomposition strategy: Chia query phức tạp thành sub-questions đơn giản.
  #
  # Ý tưởng: Query phức tạp thường khó match với documents. Chia thành
  # những questions đơn giản hơn sẽ dễ tìm được relevant information.
  #
  # Args:
  #   query (String): Query gốc phức tạp.
  #   model (String): Mô hình LLM để sử dụng.
  #
  # Returns:
  #   Array<String>: Danh sách các sub-questions.
  system_prompt = <<~HEREDOC
  Bạn là chuyên gia phân tích và decomposition questions. Nhiệm vụ của bạn là
  chia một complex query thành multiple simpler sub-questions.

  NGUYÊN TẮC DECOMPOSITION:
  1. **Atomic**: Mỗi sub-question focus vào 1 aspect cụ thể
  2. **Answerable**: Mỗi sub-question có thể được trả lời độc lập
  3. **Comprehensive**: Tất cả sub-questions cover toàn bộ original query
  4. **Logical**: Có thứ tự logic, từ basic đến advanced
  5. **Clear**: Sub-questions rõ ràng, không ambiguous

  FORMAT OUTPUT:
  Trả về danh sách sub-questions, mỗi question trên một dòng.
  KHÔNG đánh số, KHÔNG dùng bullet points, CHỈ questions thuần túy.

  EXAMPLE INPUT: "How does artificial intelligence impact healthcare and what are the challenges?"
  EXAMPLE OUTPUT:
  What is artificial intelligence in healthcare context?
  What are the main applications of AI in medical diagnosis?
  How does AI improve treatment planning and drug discovery?
  What are the technical challenges in implementing AI in hospitals?
  What are the ethical considerations for AI in healthcare?
  How does AI affect healthcare costs and accessibility?
  HEREDOC

  user_message = "Hãy decompose query này thành sub-questions:\n\n#{query}"

  response = generate_response(system_prompt, user_message, model)

  if response
    sub_questions = response.split("\n")
                          .map(&:strip)
                          .reject(&:empty?)
                          .map { |q| q.gsub(/^[\d\.\-\*\+]\s*/, '') }
                          .map(&:strip)
                          .reject(&:empty?)

    puts "✓ Tạo #{sub_questions.length} sub-questions:"
    sub_questions.each_with_index { |q, i| puts "  #{i + 1}. #{q}" }

    sub_questions
  else
    puts "❌ Không thể decompose query"
    [query]  # Fallback to original query
  end
end

def step_back_prompting(query, model = CHAT_MODEL)
  # Step-back Prompting: Tạo broader, more general question.
  #
  # Ý tưởng: Thay vì answer specific question, step back để hỏi
  # general question có liên quan. General questions thường dễ
  # match với broader documents hơn.
  #
  # Args:
  #   query (String): Query cụ thể của user.
  #   model (String): Mô hình LLM để sử dụng.
  #
  # Returns:
  #   String: Broader, more general question.
  system_prompt = <<~HEREDOC
  Bạn là chuyên gia trong việc tạo ra "step-back questions" - những câu hỏi
  broader và more general từ specific queries.

  Mục đích: Thay vì trả lời specific question, step back để hỏi general question
  liên quan. Điều này giúp tìm được broader context trước khi dive into specifics.

  NGUYÊN TẮC STEP-BACK:
  1. **Broader scope**: Question rộng hơn original query
  2. **Fundamental**: Focus vào principles, concepts cơ bản
  3. **Contextual**: Cung cấp context cho specific question
  4. **Educational**: Giúp hiểu background knowledge
  5. **Connected**: Vẫn liên quan đến original intent

  EXAMPLE INPUT: "What is the specific process for implementing OAuth 2.0 in a React application?"
  EXAMPLE OUTPUT: "What are the fundamental principles and security considerations of authentication systems in web applications?"

  FORMAT: Trả về CHỈ 1 step-back question, clear và concise.
  HEREDOC

  user_message = "Hãy tạo step-back question cho query:\n\n#{query}"

  response = generate_response(system_prompt, user_message, model)

  if response
    step_back_q = response.strip.gsub(/^[\d\.\-\*\+]\s*/, '').strip
    puts "✓ Step-back question: #{step_back_q}"
    step_back_q
  else
    puts "❌ Không thể tạo step-back question"
    query  # Fallback to original query
  end
end

def query_rewriting(query, num_variations = 3, model = CHAT_MODEL)
  # Query Rewriting: Tạo multiple paraphrases của original query.
  #
  # Ý tưởng: Cùng một ý nhưng diễn đạt khác nhau có thể match
  # với different documents. Multiple variations tăng coverage.
  #
  # Args:
  #   query (String): Query gốc.
  #   num_variations (Integer): Số variations cần tạo.
  #   model (String): Mô hình LLM để sử dụng.
  #
  # Returns:
  #   Array<String>: Danh sách query variations.
  system_prompt = <<~HEREDOC
  Bạn là chuyên gia trong việc paraphrase và rewrite queries để cải thiện
  information retrieval. Tạo ra #{num_variations} variations khác nhau của query.

  NGUYÊN TẮC REWRITING:
  1. **Semantic preservation**: Giữ nguyên ý nghĩa core
  2. **Lexical diversity**: Sử dụng từ đồng nghĩa, cách diễn đạt khác
  3. **Syntactic variation**: Thay đổi cấu trúc câu
  4. **Style diversity**: Formal/informal, direct/indirect approaches
  5. **Completeness**: Mỗi variation phải hoàn chỉnh và searchable

  FORMAT OUTPUT:
  Trả về chính xác #{num_variations} query variations, mỗi variation trên một dòng.
  KHÔNG đánh số, KHÔNG bullet points, CHỈ queries thuần túy.

  EXAMPLE INPUT: "How to improve website performance?"
  EXAMPLE OUTPUT:
  What are effective methods for optimizing web page speed?
  What techniques can enhance website loading time and responsiveness?
  How can I make my website faster and more efficient?
  HEREDOC

  user_message = "Hãy tạo #{num_variations} variations cho query:\n\n#{query}"

  response = generate_response(system_prompt, user_message, model)

  if response
    variations = response.split("\n")
                        .map(&:strip)
                        .reject(&:empty?)
                        .map { |q| q.gsub(/^[\d\.\-\*\+]\s*/, '') }
                        .map(&:strip)
                        .reject(&:empty?)
                        .first(num_variations)

    puts "✓ Tạo #{variations.length} query variations:"
    variations.each_with_index { |q, i| puts "  #{i + 1}. #{q}" }

    variations
  else
    puts "❌ Không thể rewrite query"
    [query]  # Fallback to original query
  end
end

# ===============================================================================
# Multi-Query Fusion và Ranking
# ===============================================================================

def cosine_similarity(vec1, vec2)
  # Tính cosine similarity giữa hai vector.
  v1 = Vector[*vec1]
  v2 = Vector[*vec2]

  dot_product = v1.inner_product(v2)
  magnitude1 = Math.sqrt(v1.inner_product(v1))
  magnitude2 = Math.sqrt(v2.inner_product(v2))

  return 0.0 if magnitude1 == 0.0 || magnitude2 == 0.0

  dot_product / (magnitude1 * magnitude2)
end

def multi_query_search(queries, chunks, embeddings, k_per_query = 3)
  # Thực hiện search với multiple queries và fusion kết quả.
  #
  # Sử dụng Reciprocal Rank Fusion (RRF) để combine rankings từ
  # different queries thành một final ranking.
  #
  # Args:
  #   queries (Array): Danh sách transformed queries.
  #   chunks (Array): Document chunks để search.
  #   embeddings (Array): Embeddings của chunks.
  #   k_per_query (Integer): Top-k results per query.
  #
  # Returns:
  #   Hash: {
  #     individual_results: Results per query,
  #     fused_results: Final fused ranking,
  #     rrf_scores: RRF scores for analysis
  #   }
  puts "\n--- Multi-Query Search với RRF Fusion ---"

  individual_results = {}
  all_chunk_scores = Hash.new { |h, k| h[k] = [] }

  # 1. Search với từng query
  queries.each_with_index do |query, i|
    puts "\nQuery #{i + 1}: #{query}"

    # Tạo embedding cho query
    query_embeddings = create_embeddings([query])
    next if query_embeddings.empty?

    query_embedding = query_embeddings[0]

    # Tính similarity với tất cả chunks
    similarities = embeddings.each_with_index.map do |chunk_emb, chunk_idx|
      similarity = cosine_similarity(query_embedding, chunk_emb)
      [chunk_idx, similarity]
    end

    # Sắp xếp và lấy top k
    top_results = similarities.sort_by { |_, score| -score }.first(k_per_query)
    individual_results[query] = top_results

    puts "Top #{k_per_query} results:"
    top_results.each_with_index do |(chunk_idx, score), rank|
      puts "  #{rank + 1}. Chunk #{chunk_idx}: Score #{score.round(4)}"
      all_chunk_scores[chunk_idx] << { query_idx: i, rank: rank, score: score }
    end
  end

  # 2. Reciprocal Rank Fusion (RRF)
  puts "\n--- Reciprocal Rank Fusion ---"
  rrf_scores = {}
  k_rrf = 60  # RRF parameter (standard value)

  all_chunk_scores.each do |chunk_idx, score_data|
    rrf_score = score_data.sum { |data| 1.0 / (k_rrf + data[:rank] + 1) }
    rrf_scores[chunk_idx] = {
      score: rrf_score,
      appearances: score_data.length,
      individual_scores: score_data
    }
  end

  # 3. Final ranking theo RRF scores
  fused_results = rrf_scores.sort_by { |_, data| -data[:score] }

  puts "Top 5 kết quả sau RRF fusion:"
  fused_results.first(5).each_with_index do |(chunk_idx, data), i|
    puts "#{i + 1}. Chunk #{chunk_idx}:"
    puts "   RRF Score: #{data[:score].round(4)}"
    puts "   Appeared in #{data[:appearances]}/#{queries.length} queries"
    puts "   Chunk preview: #{chunks[chunk_idx][0..100]}..."
    puts
  end

  {
    individual_results: individual_results,
    fused_results: fused_results.map { |chunk_idx, data| [chunk_idx, chunks[chunk_idx], data] },
    rrf_scores: rrf_scores
  }
end

# ===============================================================================
# Query Transform Pipeline
# ===============================================================================

def run_query_transform_pipeline(original_query, chunks, embeddings)
  # Chạy complete Query Transform pipeline với tất cả strategies.
  #
  # Pipeline:
  # 1. HyDE transformation
  # 2. Sub-question decomposition
  # 3. Step-back prompting
  # 4. Query rewriting
  # 5. Multi-query fusion search
  # 6. Comparison với basic search
  #
  # Args:
  #   original_query (String): Query gốc của user.
  #   chunks (Array): Document chunks.
  #   embeddings (Array): Embeddings của chunks.
  #
  # Returns:
  #   Hash: Complete results từ all strategies.
  puts "\n" + "=" * 80
  puts "QUERY TRANSFORM PIPELINE"
  puts "=" * 80
  puts "Original Query: #{original_query}"
  puts "=" * 80

  all_transformed_queries = [original_query]  # Include original

  # 1. HyDE Transform
  puts "\n1. HYDE TRANSFORMATION:"
  puts "-" * 50
  hyde_doc = hyde_transform(original_query)
  all_transformed_queries << hyde_doc if hyde_doc != original_query

  # 2. Sub-question Decomposition
  puts "\n2. SUB-QUESTION DECOMPOSITION:"
  puts "-" * 50
  sub_questions = sub_question_decomposition(original_query)
  all_transformed_queries.concat(sub_questions)

  # 3. Step-back Prompting
  puts "\n3. STEP-BACK PROMPTING:"
  puts "-" * 50
  step_back_q = step_back_prompting(original_query)
  all_transformed_queries << step_back_q if step_back_q != original_query

  # 4. Query Rewriting
  puts "\n4. QUERY REWRITING:"
  puts "-" * 50
  variations = query_rewriting(original_query, 3)
  all_transformed_queries.concat(variations)

  # Remove duplicates
  unique_queries = all_transformed_queries.uniq
  puts "\nTổng số unique queries: #{unique_queries.length}"

  # 5. Multi-Query Search
  puts "\n5. MULTI-QUERY SEARCH:"
  puts "-" * 50
  fusion_results = multi_query_search(unique_queries, chunks, embeddings, 3)

  # 6. Comparison với Basic Search
  puts "\n6. COMPARISON VỚI BASIC SEARCH:"
  puts "-" * 50

  # Basic search với original query
  original_embeddings = create_embeddings([original_query])
  if original_embeddings.any?
    original_embedding = original_embeddings[0]
    basic_similarities = embeddings.each_with_index.map do |chunk_emb, i|
      similarity = cosine_similarity(original_embedding, chunk_emb)
      [i, similarity, chunks[i]]
    end
    basic_results = basic_similarities.sort_by { |_, score, _| -score }.first(5)

    puts "Basic Search Results (Top 5):"
    basic_results.each_with_index do |(chunk_idx, score, chunk), i|
      puts "#{i + 1}. Chunk #{chunk_idx}: Score #{score.round(4)}"
      puts "   Preview: #{chunk[0..100]}..."
    end
  end

  puts "\nQuery Transform vs Basic Search:"
  if fusion_results[:fused_results].any?
    top_fusion = fusion_results[:fused_results].first
    fusion_chunk_idx = top_fusion[0]
    fusion_score = top_fusion[2][:score]

    puts "- Transform Top Result: Chunk #{fusion_chunk_idx} (RRF: #{fusion_score.round(4)})"
    puts "- Basic Top Result: Chunk #{basic_results.first[0]} (Cosine: #{basic_results.first[1].round(4)})"
    puts "- Khác biệt: #{fusion_chunk_idx == basic_results.first[0] ? 'GIỐNG NHAU' : 'KHÁC NHAU'}"
  end

  {
    original_query: original_query,
    all_queries: unique_queries,
    hyde_doc: hyde_doc,
    sub_questions: sub_questions,
    step_back_question: step_back_q,
    query_variations: variations,
    fusion_results: fusion_results,
    basic_results: basic_results
  }
end

# ===============================================================================
# Demo Analysis Functions
# ===============================================================================

def analyze_transformation_effectiveness(pipeline_results)
  # Phân tích hiệu quả của các transformation strategies.
  puts "\n" + "=" * 80
  puts "PHÂN TÍCH HIỆU QUẢ TRANSFORMATION"
  puts "=" * 80

  fusion_data = pipeline_results[:fusion_results]
  rrf_scores = fusion_data[:rrf_scores]

  # Phân tích query coverage
  puts "\n1. QUERY COVERAGE ANALYSIS:"
  puts "-" * 40

  total_queries = pipeline_results[:all_queries].length
  puts "Tổng số queries: #{total_queries}"

  # Chunks xuất hiện trong bao nhiêu queries
  chunk_appearances = {}
  fusion_data[:individual_results].each do |query, results|
    results.each do |chunk_idx, score|
      chunk_appearances[chunk_idx] = (chunk_appearances[chunk_idx] || 0) + 1
    end
  end

  puts "Top chunks theo số lần xuất hiện:"
  chunk_appearances.sort_by { |_, count| -count }.first(5).each do |chunk_idx, count|
    percentage = (count.to_f / total_queries * 100).round(1)
    puts "  Chunk #{chunk_idx}: #{count}/#{total_queries} queries (#{percentage}%)"
  end

  # Phân tích diversity của results
  puts "\n2. RESULT DIVERSITY ANALYSIS:"
  puts "-" * 40

  unique_chunks_found = chunk_appearances.keys.length
  total_chunks = pipeline_results[:fusion_results][:fused_results].length

  puts "Unique chunks found: #{unique_chunks_found}"
  puts "Total chunks in final ranking: #{total_chunks}"

  # Strategy effectiveness
  puts "\n3. STRATEGY EFFECTIVENESS:"
  puts "-" * 40

  strategies = {
    'Original Query' => [pipeline_results[:original_query]],
    'HyDE' => [pipeline_results[:hyde_doc]],
    'Sub-questions' => pipeline_results[:sub_questions],
    'Step-back' => [pipeline_results[:step_back_question]],
    'Variations' => pipeline_results[:query_variations]
  }

  strategies.each do |strategy_name, queries|
    next if queries.nil? || queries.empty?

    puts "\n#{strategy_name}:"
    puts "  Number of queries: #{queries.length}"

    if strategy_name == 'Sub-questions' && queries.length > 1
      puts "  Average query length: #{(queries.map(&:length).sum.to_f / queries.length).round(1)} chars"
    end
  end
end

# ===============================================================================
# Demo chính
# ===============================================================================

def run_query_transform_demo
  # Chạy demo hoàn chỉnh Query Transform RAG.
  puts "\n=== Demo Query Transform RAG ==="

  # Bước 1: Load và chuẩn bị data
  puts "\n1. Chuẩn bị dữ liệu..."
  pdf_path = "data/AI_Information.pdf"
  extracted_text = extract_text_from_pdf(pdf_path)
  chunks = chunk_text(extracted_text, 1000, 200)

  # Giới hạn chunks cho demo
  demo_chunks = chunks.first(10)
  puts "Demo với #{demo_chunks.length} chunks"

  # Tạo embeddings
  puts "Tạo embeddings cho chunks..."
  embeddings = create_embeddings(demo_chunks)
  return unless embeddings.any?

  # Bước 2: Test với multiple queries
  test_queries = [
    "What are the practical applications of machine learning in healthcare diagnosis?",
    "How can businesses implement AI solutions cost-effectively?",
    "What are the risks and ethical considerations of artificial intelligence?",
    "Explain the process of training deep learning neural networks"
  ]

  # Bước 3: Chạy pipeline cho mỗi query
  test_queries.each_with_index do |query, i|
    puts "\n" + "=" * 100
    puts "DEMO #{i + 1}: Query Transform Pipeline"
    puts "=" * 100

    # Chạy complete pipeline
    pipeline_results = run_query_transform_pipeline(query, demo_chunks, embeddings)

    # Phân tích effectiveness
    analyze_transformation_effectiveness(pipeline_results)

    puts "\n" + "=" * 100
  end
end

# ===============================================================================
# Demo Strategy Comparison
# ===============================================================================

def compare_transformation_strategies
  # So sánh trực tiếp các transformation strategies.
  puts "\n=== So sánh Transformation Strategies ==="

  # Load data
  pdf_path = "data/AI_Information.pdf"
  extracted_text = extract_text_from_pdf(pdf_path)
  chunks = chunk_text(extracted_text, 800, 150).first(5)  # Nhỏ hơn cho demo
  embeddings = create_embeddings(chunks)
  return unless embeddings.any?

  test_query = "How does AI improve medical diagnosis accuracy?"
  puts "Test Query: #{test_query}"
  puts "=" * 60

  # Test từng strategy độc lập
  strategies = {
    'Original' => [test_query],
    'HyDE' => [hyde_transform(test_query)],
    'Sub-questions' => sub_question_decomposition(test_query),
    'Step-back' => [step_back_prompting(test_query)],
    'Variations' => query_rewriting(test_query, 2)
  }

  puts "\n" + "=" * 60
  puts "STRATEGY COMPARISON RESULTS"
  puts "=" * 60

  strategies.each do |strategy_name, queries|
    next if queries.nil? || queries.empty?

    puts "\n#{strategy_name.upcase} STRATEGY:"
    puts "-" * 30

    # Search với queries từ strategy này
    all_results = []
    queries.each_with_index do |q, i|
      next if q.nil? || q.empty?

      puts "Query #{i + 1}: #{q[0..80]}#{'...' if q.length > 80}"

      # Tìm kiếm
      q_embeddings = create_embeddings([q])
      next if q_embeddings.empty?

      q_embedding = q_embeddings[0]
      similarities = embeddings.each_with_index.map do |chunk_emb, chunk_idx|
        similarity = cosine_similarity(q_embedding, chunk_emb)
        [chunk_idx, similarity]
      end

      top_result = similarities.max_by { |_, score| score }
      all_results << top_result
      puts "  Best match: Chunk #{top_result[0]} (Score: #{top_result[1].round(4)})"
    end

    # Summary cho strategy
    if all_results.any?
      avg_score = (all_results.map { |_, score| score }.sum / all_results.length).round(4)
      unique_chunks = all_results.map { |chunk_idx, _| chunk_idx }.uniq.length
      puts "Strategy Summary:"
      puts "  Average top score: #{avg_score}"
      puts "  Unique chunks found: #{unique_chunks}"
      puts "  Total queries: #{queries.length}"
    end
  end
end

# ===============================================================================
# Main Execution
# ===============================================================================

if __FILE__ == $0
  puts "Query Transform RAG bằng Ruby"
  puts "=" * 60

  begin
    # Demo chính với complete pipeline
    run_query_transform_demo

    # So sánh strategies
    puts "\n\n"
    compare_transformation_strategies

    puts "\n=== HOÀN THÀNH DEMO QUERY TRANSFORM RAG ==="
  rescue => e
    puts "Lỗi trong quá trình thực thi: #{e.message}"
    puts e.backtrace
  end
end
