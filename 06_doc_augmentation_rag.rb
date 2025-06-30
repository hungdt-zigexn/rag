#!/usr/bin/env ruby
# frozen_string_literal: true

# ===============================================================================
# Document Augmentation RAG (Phiên bản Ruby)
# ===============================================================================
#
# Document Augmentation RAG giải quyết vấn đề "Semantic Mismatch" - khi
# user query và text content nói về cùng một chủ đề nhưng sử dụng từ ngữ
# khác nhau, dẫn đến similarity thấp và retrieval thất bại.
#
# Kỹ thuật này tạo ra các câu hỏi giả (synthetic questions) cho mỗi chunk,
# rồi sử dụng những câu hỏi này làm "cầu nối ngữ nghĩa" để cải thiện retrieval.
#
# Quy trình:
# 1. Trích xuất văn bản từ PDF.
# 2. Chia thành các chunk nhỏ.
# 3. Tạo multiple synthetic questions cho mỗi chunk.
# 4. Kết hợp questions với chunk content.
# 5. Tạo embeddings cho augmented chunks.
# 6. Thực hiện RAG với enhanced semantic matching.

require 'net/http'
require 'uri'
require 'json'
require 'matrix'
require 'pdf-reader'

# ===============================================================================
# Thiết lập Môi trường
# ===============================================================================
puts "=== Document Augmentation RAG ==="

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
# Hàm Trích xuất và Chia nhỏ
# ===============================================================================

def extract_text_from_pdf(pdf_path)
  # Trích xuất văn bản từ file PDF.
  #
  # Args:
  #   pdf_path (String): Đường dẫn đến file PDF.
  #
  # Returns:
  #   String: Văn bản được trích xuất từ PDF.
  reader = PDF::Reader.new(pdf_path)
  all_text = ""

  reader.pages.each do |page|
    all_text += page.text + " "
  end

  all_text.strip
end

def chunk_text(text, chunk_size = 1000, overlap = 200)
  # Chia văn bản thành các chunk có overlap.
  #
  # Args:
  #   text (String): Văn bản cần chia nhỏ.
  #   chunk_size (Integer): Số ký tự trong mỗi chunk.
  #   overlap (Integer): Số ký tự overlap giữa các chunk.
  #
  # Returns:
  #   Array<String>: Mảng các chunk văn bản.
  chunks = []
  step_size = chunk_size - overlap

  (0...text.length).step(step_size) do |i|
    chunk = text[i, chunk_size]
    chunks << chunk unless chunk.empty?
  end

  chunks
end

# ===============================================================================
# Hàm API Client
# ===============================================================================

def make_api_request(endpoint, payload)
  # Thực hiện API request đến OpenAI-compatible endpoint.
  #
  # Args:
  #   endpoint (String): Đường dẫn API.
  #   payload (Hash): Dữ liệu gửi đi.
  #
  # Returns:
  #   Hash: Dữ liệu trả về từ API.
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
  #
  # Args:
  #   text_list (Array): Mảng văn bản đầu vào.
  #   model (String): Mô hình embedding.
  #
  # Returns:
  #   Array: Mảng embeddings.
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
  #
  # Args:
  #   system_prompt (String): Hệ thống prompt.
  #   user_message (String): Câu hỏi của người dùng.
  #   model (String): Mô hình LLM.
  #
  # Returns:
  #   String: Phản hồi từ mô hình AI.
  payload = {
    model: model,
    temperature: 0.7,  # Slight creativity for question generation
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
# Kỹ thuật chính: Document Augmentation với Synthetic Questions
# ===============================================================================

def generate_questions(text_chunk, num_questions = 5, model = CHAT_MODEL)
  # Tạo synthetic questions cho một chunk văn bản.
  #
  # Đây là kỹ thuật cốt lõi của Document Augmentation - tạo ra nhiều
  # câu hỏi khác nhau mà chunk này có thể trả lời, giúp tăng khả năng
  # match với diverse user queries.
  #
  # Args:
  #   text_chunk (String): Nội dung chunk cần tạo questions.
  #   num_questions (Integer): Số lượng câu hỏi cần tạo.
  #   model (String): Mô hình LLM để sử dụng.
  #
  #
  # Returns:
  #   Array<String>: Danh sách các câu hỏi được tạo.
  #
  system_prompt = "
  Bạn là một chuyên gia trong việc tạo ra các câu hỏi đa dạng và chất lượng cao.
  Nhiệm vụ của bạn là tạo ra #{num_questions} câu hỏi khác nhau mà đoạn văn bản
  được cung cấp có thể trả lời.

  HƯỚNG DẪN QUAN TRỌNG:
  1. **Đa dạng về góc độ:** Tạo câu hỏi từ nhiều góc độ khác nhau (what, how, why, when, where)
  2. **Đa dạng về mức độ:** Từ câu hỏi cơ bản đến phức tạp, từ cụ thể đến trừu tượng
  3. **Đa dạng về từ vựng:** Sử dụng từ đồng nghĩa, cách diễn đạt khác nhau
  4. **Relevant và answerable:** Đảm bảo đoạn văn có thể trả lời được câu hỏi
  5. **Natural language:** Câu hỏi nghe tự nhiên như người thật hỏi

  FORMAT OUTPUT:
  Trả về chính xác #{num_questions} câu hỏi, mỗi câu trên một dòng.
  KHÔNG đánh số, KHÔNG dùng bullet points, CHỈ câu hỏi thuần túy.

  VÍ DỤ TỐT:
  - \"What are the main applications of machine learning in healthcare?\"
  - \"How does artificial intelligence improve medical diagnosis accuracy?\"
  - \"Which AI technologies are most promising for personalized medicine?\"
  - \"What challenges exist in implementing AI solutions in hospitals?\"
  - \"How can machine learning help reduce healthcare costs?\"
  "

  user_message = "
  Hãy tạo #{num_questions} câu hỏi đa dạng cho đoạn văn bản sau:

  #{text_chunk}

  Câu hỏi:
  "

  response = generate_response(system_prompt, user_message, model)

  if response
    # Parse questions từ response
    questions = response.split("\n")
                       .map(&:strip)
                       .reject(&:empty?)
                       .map { |q| q.gsub(/^[\d\.\-\*\+]\s*/, '') } # Remove numbering
                       .map(&:strip)
                       .reject(&:empty?)
                       .first(num_questions) # Ensure we get exactly num_questions

    puts "Generated #{questions.length} questions for chunk"
    questions.each_with_index do |q, i|
      puts "  #{i + 1}. #{q}"
    end

    questions
  else
    puts "Failed to generate questions for chunk"
    []
  end
end

def create_augmented_chunks(text_chunks, questions_per_chunk = 5)
  # Tạo augmented chunks bằng cách thêm synthetic questions.
  #
  # Quy trình Document Augmentation:
  # 1. Lặp qua từng chunk gốc
  # 2. Tạo multiple synthetic questions cho chunk đó
  # 3. Kết hợp questions với original content
  # 4. Tạo augmented chunk với enhanced semantic coverage
  #
  # Args:
  #   text_chunks (Array): Mảng các chunk văn bản gốc.
  #   questions_per_chunk (Integer): Số câu hỏi tạo cho mỗi chunk.
  #
  # Returns:
  #   Hash: {
  #     augmented_chunks: Array - Chunks đã được augment,
  #     questions_map: Hash - Mapping từ chunk index đến questions,
  #     original_chunks: Array - Chunks gốc
  #   }
  puts "\n--- Bắt đầu Document Augmentation ---"
  puts "Tạo #{questions_per_chunk} synthetic questions cho mỗi chunk..."

  augmented_chunks = []
  questions_map = {}
  total_chunks = text_chunks.length

  text_chunks.each_with_index do |chunk, index|
    puts "\n" + "=" * 60
    puts "Xử lý chunk #{index + 1}/#{total_chunks}"
    puts "=" * 60

    # Hiển thị preview của chunk
    puts "Chunk content preview:"
    puts chunk[0..200] + (chunk.length > 200 ? "..." : "")
    puts

    # Tạo synthetic questions cho chunk
    puts "Generating synthetic questions..."
    questions = generate_questions(chunk, questions_per_chunk)
    questions_map[index] = questions

    if questions.any?
      # Tạo augmented chunk với questions
      questions_section = questions.map { |q| "- #{q}" }.join("\n")

      augmented_chunk = "
SYNTHETIC QUESTIONS:
#{questions_section}

CONTENT:
#{chunk}
"
      augmented_chunks << augmented_chunk
      puts "✓ Chunk #{index + 1} augmented with #{questions.length} questions"
    else
      # Fallback: sử dụng chunk gốc nếu không tạo được questions
      puts "⚠ Failed to generate questions for chunk #{index + 1}, using original"
      augmented_chunks << chunk
      questions_map[index] = []
    end

    # Delay để tránh rate limiting
    sleep(1)
  end

  puts "\n--- Hoàn thành Document Augmentation ---"
  total_questions = questions_map.values.sum(&:length)
  puts "Tổng số questions tạo thành công: #{total_questions}"
  puts "Trung bình questions per chunk: #{(total_questions.to_f / total_chunks).round(2)}"

  {
    augmented_chunks: augmented_chunks,
    questions_map: questions_map,
    original_chunks: text_chunks
  }
end

# ===============================================================================
# Hàm Tìm kiếm Semantic
# ===============================================================================

def cosine_similarity(vec1, vec2)
  # Tính cosine similarity giữa hai vector.
  #
  # Args:
  #   vec1 (Array): Vector thứ nhất.
  #   vec2 (Array): Vector thứ hai.
  #
  # Returns:
  #   Float: Độ tương tự cosine giữa hai vector.
  v1 = Vector[*vec1]
  v2 = Vector[*vec2]

  dot_product = v1.inner_product(v2)
  magnitude1 = Math.sqrt(v1.inner_product(v1))
  magnitude2 = Math.sqrt(v2.inner_product(v2))

  return 0.0 if magnitude1 == 0.0 || magnitude2 == 0.0

  dot_product / (magnitude1 * magnitude2)
end

def semantic_search(query, chunks, embeddings, k = 5, chunk_type = "basic")
  # Tìm kiếm semantic với enhanced logging.
  #
  # Args:
  #   query (String): Câu hỏi của người dùng.
  #   chunks (Array): Chunks để search.
  #   embeddings (Array): Embeddings của chunks.
  #   k (Integer): Số chunks cần trả về.
  #   chunk_type (String): Loại chunks ("basic" hoặc "augmented").
  #
  # Returns:
  #   Array: Top-k chunks liên quan nhất.
  # Tạo embedding cho query
  query_embeddings = create_embeddings([query])
  return [] if query_embeddings.empty?

  query_embedding = query_embeddings[0]

  # Tính similarity scores
  similarities = embeddings.each_with_index.map do |chunk_emb, i|
    similarity = cosine_similarity(query_embedding, chunk_emb)
    [i, similarity, chunks[i]]
  end

  # Sắp xếp và lấy top k
  top_results = similarities.sort_by { |_, score, _| -score }.first(k)

  puts "\nTop #{k} kết quả tìm kiếm (#{chunk_type}):"
  top_results.each_with_index do |(index, score, chunk), i|
    # Extract preview từ chunk
    if chunk_type == "augmented"
      content_start = chunk.index("CONTENT:")
      preview = content_start ? chunk[content_start + 8..content_start + 108] : chunk[0..100]
    else
      preview = chunk[0..100]
    end

    puts "#{i + 1}. Score: #{score.round(4)} - #{preview.strip}..."
  end

  top_results.map { |_, _, chunk| chunk }
end

# ===============================================================================
# So sánh Basic RAG vs Document Augmentation RAG
# ===============================================================================

def compare_basic_vs_augmented(query, original_chunks, augmented_chunks,
                               original_embeddings, augmented_embeddings)
  # So sánh kết quả giữa Basic RAG và Document Augmentation RAG.
  #
  # Args:
  #   query (String): Câu hỏi của người dùng.
  #   original_chunks (Array): Chunks gốc.
  #   augmented_chunks (Array): Chunks đã được augment.
  #   original_embeddings (Array): Embeddings của chunks gốc.
  #   augmented_embeddings (Array): Embeddings của chunks đã được augment.
  #
  puts "\n=== So sánh Basic RAG vs Document Augmentation RAG ==="
  puts "Query: #{query}"
  puts "=" * 80

  # 1. Basic RAG (không có synthetic questions)
  puts "\n1. BASIC RAG (KHÔNG CÓ SYNTHETIC QUESTIONS):"
  puts "-" * 60
  basic_results = semantic_search(query, original_chunks, original_embeddings, 3, "basic")

  if basic_results.any?
    puts "\nTop chunk (Basic RAG):"
    puts basic_results[0][0..300] + "..."
    puts "Độ dài: #{basic_results[0].length} ký tự"
  end

  # 2. Document Augmentation RAG (có synthetic questions)
  puts "\n2. DOCUMENT AUGMENTATION RAG (CÓ SYNTHETIC QUESTIONS):"
  puts "-" * 60
  augmented_results = semantic_search(query, augmented_chunks, augmented_embeddings, 3, "augmented")

  if augmented_results.any?
    puts "\nTop chunk (Document Augmentation RAG):"

    # Extract và hiển thị synthetic questions
    top_chunk = augmented_results[0]
    questions_end = top_chunk.index("CONTENT:")
    if questions_end
      questions_section = top_chunk[0...questions_end]
      content_section = top_chunk[questions_end + 8..-1]

      puts "\nSynthetic Questions trong chunk này:"
      puts questions_section.strip

      puts "\nContent preview:"
      puts content_section[0..300] + "..."
    else
      puts top_chunk[0..400] + "..."
    end

    puts "Độ dài total: #{top_chunk.length} ký tự"
  end

  puts "=" * 80

  { basic: basic_results, augmented: augmented_results }
end

# ===============================================================================
# Tạo phản hồi với Augmented Chunks
# ===============================================================================

def generate_augmented_response(query, augmented_chunks)
  # Tạo phản hồi AI sử dụng augmented chunks.
  #
  # Args:
  #   query (String): Câu hỏi của người dùng.
  #   augmented_chunks (Array): Chunks đã được augment.
  #
  # Returns:
  #   String: Phản hồi từ mô hình AI.
  system_prompt = "
  Bạn là một AI assistant thông minh với khả năng xử lý văn bản đã được augment.
  Bạn sẽ nhận được các đoạn văn bản có kèm theo \"SYNTHETIC QUESTIONS\" và \"CONTENT\".

  CHÚ Ý:
  - SYNTHETIC QUESTIONS giúp bạn hiểu những gì văn bản có thể trả lời
  - CONTENT chứa thông tin thực tế để trả lời câu hỏi
  - Sử dụng cả hai phần để hiểu đầy đủ ngữ cảnh và khả năng của văn bản
  - Ưu tiên sử dụng CONTENT để trả lời, nhưng dùng QUESTIONS để hiểu relevance
  - Nếu thông tin không đủ, hãy nói rõ những gì còn thiếu
  "

  # Chuẩn bị context từ augmented chunks
  context = augmented_chunks.each_with_index.map do |chunk, i|
    "=== AUGMENTED CONTEXT #{i + 1} ===\n#{chunk}"
  end.join("\n\n")

  user_message = "#{context}\n\n=== CÂU HỎI ===\n#{query}"

  generate_response(system_prompt, user_message, CHAT_MODEL)
end

# ===============================================================================
# Demo chính
# ===============================================================================

def run_document_augmentation_demo
  # Chạy demo hoàn chỉnh Document Augmentation RAG.
  #
  # Args:
  #   None
  #
  # Returns:
  #   None
  puts "\n=== Demo Document Augmentation RAG ==="

  # Bước 1: Trích xuất văn bản từ PDF
  puts "\n1. Trích xuất văn bản từ PDF..."
  pdf_path = "data/AI_Information.pdf"
  extracted_text = extract_text_from_pdf(pdf_path)
  puts "Độ dài văn bản: #{extracted_text.length} ký tự"

  # Bước 2: Chia thành chunks
  puts "\n2. Chia văn bản thành chunks..."
  original_chunks = chunk_text(extracted_text, 1200, 200)
  puts "Số chunks được tạo: #{original_chunks.length}"

  # Giới hạn số chunks để demo (tránh quá nhiều API calls)
  demo_chunks = original_chunks.first(3)
  puts "Demo với #{demo_chunks.length} chunks đầu tiên"

  # Bước 3: Tạo document augmentation
  puts "\n3. Thực hiện Document Augmentation..."
  augmented_data = create_augmented_chunks(demo_chunks, 4)
  augmented_chunks = augmented_data[:augmented_chunks]
  questions_map = augmented_data[:questions_map]

  puts "\nTóm tắt Synthetic Questions được tạo:"
  questions_map.each do |chunk_index, questions|
    puts "Chunk #{chunk_index + 1}: #{questions.length} questions"
    questions.first(2).each { |q| puts "  - #{q}" }
    puts "  ..." if questions.length > 2
  end

  # Bước 4: Tạo embeddings cho cả original và augmented chunks
  puts "\n4. Tạo embeddings..."
  puts "Tạo embeddings cho original chunks..."
  original_embeddings = create_embeddings(demo_chunks)

  puts "Tạo embeddings cho augmented chunks..."
  augmented_embeddings = create_embeddings(augmented_chunks)

  return unless original_embeddings.any? && augmented_embeddings.any?

  # Bước 5: So sánh với nhiều queries khác nhau
  test_queries = [
    "Các ứng dụng thực tế của AI trong chăm sóc sức khỏe là gì?",
    "Machine learning có thể cải thiện hoạt động kinh doanh như thế nào?",
    "Những tác động đạo đức của trí tuệ nhân tạo là gì?",
    "AI giúp ích gì trong chẩn đoán và điều trị y tế?"
  ]

  test_queries.each_with_index do |query, i|
    puts "\n" + "=" * 100
    puts "DEMO #{i + 1}: #{query}"
    puts "=" * 100

    # So sánh Basic vs Document Augmentation
    comparison_results = compare_basic_vs_augmented(
      query,
      demo_chunks,
      augmented_chunks,
      original_embeddings,
      augmented_embeddings
    )

    # Tạo phản hồi với Document Augmentation
    if comparison_results[:augmented].any?
      puts "\n3. PHẢN HỒI VỚI DOCUMENT AUGMENTATION:"
      puts "-" * 60
      augmented_response = generate_augmented_response(query, comparison_results[:augmented])

      if augmented_response
        puts augmented_response
      else
        puts "Không thể tạo phản hồi"
      end
    end

    puts "\n" + "=" * 100
  end
end

# ===============================================================================
# Demo phân tích Synthetic Questions
# ===============================================================================

def analyze_question_quality
  # Demo và phân tích chất lượng của synthetic questions.
  #
  # Args:
  #   None
  #
  # Returns:
  #   None
  puts "\n=== Phân tích chất lượng Synthetic Questions ==="

  # Load một chunk mẫu để phân tích
  pdf_path = "data/AI_Information.pdf"
  extracted_text = extract_text_from_pdf(pdf_path)
  sample_chunk = chunk_text(extracted_text, 1000, 150).first

  puts "Phân tích chất lượng questions cho chunk mẫu:\n"
  puts "=" * 60
  puts "CHUNK CONTENT:"
  puts "=" * 60
  puts sample_chunk[0..400] + "..."
  puts

  # Tạo questions với số lượng khác nhau
  [3, 5, 7].each do |num_questions|
    puts "\n" + "-" * 50
    puts "TESTING với #{num_questions} questions:"
    puts "-" * 50

    questions = generate_questions(sample_chunk, num_questions)

    if questions.any?
      puts "\nPhân tích chất lượng:"
      puts "- Số questions thực tế: #{questions.length}"
      puts "- Độ dài trung bình: #{(questions.map(&:length).sum.to_f / questions.length).round(1)} ký tự"

      # Phân tích diversity
      question_words = questions.map { |q| q.downcase.split }.flatten
      unique_words = question_words.uniq.length
      total_words = question_words.length
      diversity_ratio = (unique_words.to_f / total_words * 100).round(1)

      puts "- Diversity ratio: #{diversity_ratio}% (unique words / total words)"

      # Phân tích question types
      question_types = questions.map do |q|
        case q.downcase
        when /^what/ then "What"
        when /^how/ then "How"
        when /^why/ then "Why"
        when /^when/ then "When"
        when /^where/ then "Where"
        when /^which/ then "Which"
        when /^who/ then "Who"
        else "Other"
        end
      end

      type_counts = question_types.group_by(&:itself).transform_values(&:count)
      puts "- Question types: #{type_counts}"
    else
      puts "❌ Không thể tạo questions"
    end
  end
end

# ===============================================================================
# Main Execution
# ===============================================================================

if __FILE__ == $0
  puts "Document Augmentation RAG bằng Ruby"
  puts "=" * 60

  begin
    # Demo chính
    run_document_augmentation_demo

    # Phân tích question quality
    puts "\n\n"
    analyze_question_quality

    puts "\n=== HOÀN THÀNH DEMO DOCUMENT AUGMENTATION RAG ==="
  rescue => e
    puts "Lỗi trong quá trình thực thi: #{e.message}"
    puts e.backtrace
  end
end
