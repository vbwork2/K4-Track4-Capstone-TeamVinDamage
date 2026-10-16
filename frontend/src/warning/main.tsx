import * as React from 'react';
import {createRoot} from 'react-dom/client';
import Dashboard from './Dashboard';
const root=document.getElementById('root');
if(!root)throw new Error('Expected #root');
createRoot(root).render(<React.StrictMode><Dashboard/></React.StrictMode>);
