package misc;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;

import com.mongodb.BasicDBList;
import com.mongodb.DBObject;
import com.mongodb.util.JSON;


public class YandexTranslator {
	private int queries = 0;
	private String API_KEY;
	public YandexTranslator() {
		this("trnsl.1.1.20160229T101024Z.da9ea733a547cc2d.89c054266065f4d620677c011fa9820a1c8a18d6");
	}
	public YandexTranslator(String API_KEY) {
		this.API_KEY = API_KEY;
	}
	public String translate(String text) throws Exception {
		if(text.isEmpty())
			return text;
		//generate query
		String query = "";
		for(String str : text.split("\\s+")) {
			if(!query.isEmpty())
				query += "+";
			query += str.trim();
		}
		queries++;
		//System.out.println("Query #"+queries+" (length "+query.length()+")");
		query = "https://translate.yandex.net/api/v1.5/tr.json/translate?key="+API_KEY+"&lang=en&text="+query;
		String result = "";
		int errorCode = 200;//success code is default
		try{
			String response = getHTML(new URL(query));
			DBObject dbObject = (DBObject) JSON.parse(response);
			//System.out.println("Query "+queries);
			BasicDBList list = (BasicDBList) dbObject.get("text");
			for(int i=0;i<list.size();i++)
				for(String str : ((String)list.get(i)).split("\\s+"))
					result += str + " ";
			errorCode = (int) dbObject.get("code");
		}
		catch(Exception e) {
			e.printStackTrace();
		} 
		switch(errorCode) {
		case 200:break;//correct operation
		case 402:throw new Exception("Blocked key: "+API_KEY);
		case 403:throw new Exception("Daily number of requests exceeded for key: "+API_KEY);
		case 404:throw new Exception("Daily size of translated text exceeded for key: "+API_KEY);
		case 413:throw new Exception("Exceeded maximum text size");
		case 422:throw new Exception("Failed to translate: "+text);
		case 501:throw new Exception("Translation direction not supported");
		default: throw new Exception("Unknown translation error");
		}
		return result;
	}
	public int getNumberOfSentQueries() {
		return queries;
	}
	public int getRequestSizeLimit() {
		return 9800;
	}
	

	/**
	 * <h1>getHTML</h1>
	 * @param url a given URL - e.g <code>new URL(addressString)</code>
	 * @return String representation of the GET request on the given URL
	 * (this is usually plain text HTML and can later be converted to other specified formats, such as JSON)
	 * @throws Exception
	 */
	public static String getHTML(URL url) {
		try{
		  StringBuilder result = new StringBuilder();
		  HttpURLConnection conn = (HttpURLConnection) url.openConnection();
		  conn.setRequestMethod("GET");
		  BufferedReader rd = new BufferedReader(new InputStreamReader(conn.getInputStream()));
		  String line;
		  while ((line = rd.readLine()) != null) {
		     result.append(line);
		  }
		  rd.close();
		  return result.toString();
		}
		catch(Exception e) {
			System.err.println(e.toString());
			return "";
		}
	}
}
