import React, { useEffect, useState, useRef  } from 'react';
import axios from 'axios';
import Link from 'next/link'
import { GameOfDay } from './components/game-of-the-day'
import { PredictionList } from './components/prediction-list'
import { PastPredictions } from './components/past-predictions'
import { NBANews } from './components/nba-news'
import './globals.css';
import { NBATeamName } from '../src/types/nba-teams';



interface GameData {
  id: number;
  homeTeam: NBATeamName;
  homeRecord: string;
  awayTeam: NBATeamName;
  awayRecord: string;
  prediction: NBATeamName;
  confidence: number;
}


interface GameOfDayData {
  homeTeam: {
    name: NBATeamName;
    record: string;
    winChance: number;
  };
  awayTeam: {
    name: NBATeamName;
    record: string;
    winChance: number;
  };
  gameTime: string;
  stadium: string;
}


interface ApiResponse {
  message: string;
  games: GameData[];
  gameOfDay: GameOfDayData;
}


const pastPredictionsData: {
  id: number;
  homeTeam: NBATeamName;
  awayTeam: NBATeamName;
  prediction: NBATeamName;
  actual: NBATeamName;
}[] = [
  { id: 1, homeTeam: 'RAPTORS', awayTeam: 'SIXERS', prediction: 'RAPTORS', actual: 'SIXERS' },
  { id: 2, homeTeam: 'NUGGETS', awayTeam: 'JAZZ', prediction: 'NUGGETS', actual: 'NUGGETS'},
  { id: 3, homeTeam: 'CELTICS', awayTeam: 'BUCKS', prediction: 'BUCKS', actual: 'CELTICS'},
  { id: 4, homeTeam: 'WARRIORS', awayTeam: 'SUNS', prediction: 'WARRIORS', actual: 'WARRIORS'},
];


// const predictionListData: {
//   id: number;
//   homeTeam: NBATeamName;
//   homeRecord: string;
//   awayTeam: NBATeamName;
//   awayRecord: string;
//   prediction: NBATeamName;
//   confidence: number;
// }[] = [
//   { id: 1, homeTeam: 'WARRIORS', homeRecord: '25-16', awayTeam: 'SUNS', awayRecord: '23-18', prediction: 'WARRIORS', confidence: 58 },
//   { id: 2, homeTeam: 'BUCKS', homeRecord: '27-14', awayTeam: 'NETS', awayRecord: '24-17', prediction: 'BUCKS', confidence: 62 },
//   { id: 3, homeTeam: 'HEAT', homeRecord: '22-19', awayTeam: 'KNICKS', awayRecord: '21-20', prediction: 'HEAT', confidence: 55 },
//   { id: 4, homeTeam: 'SIXERS', homeRecord: '26-15', awayTeam: 'BULLS', awayRecord: '20-21', prediction: 'SIXERS', confidence: 70 },
// ];

const App: React.FC = () => {
  const [data, setData] = useState<ApiResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get<ApiResponse>('http://127.0.0.1:5000/api/data');
        setData(response.data);
        setLoading(false);
      } catch (err) {
        setError('Error fetching data');
        setLoading(false);
        console.error("There was an error!", err);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 60000);
    
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return <div className="text-center p-4">Loading...</div>;
  }

  if (error) {
    return <div className="text-center p-4 text-red-500">{error}</div>;
  }

  return (

    <div>
    
              <div className="min-h-screen bg-gray-100">
        <header className="bg-gradient-to-r from-blue-800 to-red-700 text-white p-4 shadow-lg">
          <nav className="container mx-auto flex flex-wrap justify-between items-center">
            <h1 className="text-3xl font-bold">NBA Predictor</h1>
          </nav>
        </header>
  
        <main className="container mx-auto p-4">
          {data?.gameOfDay && <GameOfDay {...data.gameOfDay} />}
          
          <div className=" gap-6 mt-8">
            {data?.games && <PredictionList predictions={data.games} />}
          </div>
  
          <NBANews />
        </main>
  
        <footer className="bg-gradient-to-r from-blue-800 to-red-700 text-white p-6 mt-8">
          <div className="container mx-auto text-center">
            <p>© 2025 NBA Predictor. All Rights Reserved.</p>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default App;
