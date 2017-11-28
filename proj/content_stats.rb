#!/usr/bin/env ruby
require "thor"
require "ptools"
require "httparty"

class ContentStats < Thor

  desc "describe", "Describe WootMath AP Content"
  def describe
    url = 'http://10.0.10.141/woot_cms/v1.0/user/release/module/wootmath_fractions.json/print'
    response = HTTParty.get(url, format: :plain)
    data = JSON.parse response, symbolize_names: true
    #puts JSON.pretty_generate(data)
    l_type = data[:children].inject(0)  { |memo, c| memo += 1 if (c[:type] == 'lesson' && c[:meta][:lesson_type] == 'l'); memo }
    s_type = data[:children].inject(0)  { |memo, c| memo += 1 if (c[:type] == 'lesson' && c[:meta][:lesson_type] == 's'); memo }
    i_type = data[:children].inject(0)  { |memo, c| memo += 1 if (c[:type] == 'lesson' && c[:meta][:lesson_type] == 'i'); memo }
    b_type = data[:children].inject(0)  { |memo, c| memo += 1 if (c[:type] == 'lesson' && c[:meta][:lesson_type] == 'i'); memo }
    n_books = data[:children].inject(0)  { |memo, c| memo += 1 if (c[:type] == 'book' && c[:group] != 'settings'); memo }
    puts "N l-type lessons #{l_type}"
    puts "N s-type lessons #{s_type}"
    puts "N i-type lessons #{i_type}"
    puts "N book #{n_books}"
  end

  default_task :describe

end

ContentStats.start

