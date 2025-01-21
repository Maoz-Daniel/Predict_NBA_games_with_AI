export function NBANews() {
    const insightItems = [
      { 
        id: 1, 
        title: "The Power of Advanced Analytics", 
        category: "Data Science", 
        description: "How machine learning algorithms process thousands of data points to predict game outcomes with increasing accuracy.",
        icon: "📊"
      },
      { 
        id: 2, 
        title: "Home Court Advantage Decoded", 
        category: "Statistical Analysis", 
        description: "Unveiling the hidden factors that make playing at home a game-changer in the NBA.",
        icon: "🏟️"
      },
      { 
        id: 3, 
        title: "The Injury Impact Model", 
        category: "Predictive Modeling", 
        description: "Exploring our unique approach to quantifying how player injuries affect team performance and game predictions.",
        icon: "🏥"
      },
      { 
        id: 4, 
        title: "Momentum: Myth or Math?", 
        category: "Trend Analysis", 
        description: "Investigating whether a team's recent performance streak is a reliable predictor of future games.",
        icon: "📈"
      },
    ]
  
    return (
      <div className="mt-8 bg-white p-6 rounded-lg shadow-md border-t-4 border-blue-600">
        <h3 className="text-2xl font-bold text-blue-900 mb-6">Prediction Insights</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {insightItems.map((item) => (
            <div key={item.id} className="bg-gray-50 rounded-lg p-4 shadow-sm hover:shadow-md transition-shadow duration-300">
              <div className="flex items-center mb-2">
                <span className="text-3xl mr-3">{item.icon}</span>
                <h4 className="font-semibold text-lg text-blue-800">{item.title}</h4>
              </div>
              <p className="text-sm text-gray-600 mb-2">{item.category}</p>
              <p className="text-sm text-gray-700">{item.description}</p>
            </div>
          ))}
        </div>
      </div>
    )
  }
  
  